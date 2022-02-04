import argparse
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from allennlp.data import Vocabulary

from updown.config import Config
from updown.data.datasets import EvaluationDataset, EvaluationDatasetWithConstraints
from var_updown.models import UpDownCaptioner
from updown.utils.constraints import add_constraint_words_to_vocabulary


parser = argparse.ArgumentParser(
    "Run inference using UpDown Captioner, on either nocaps val or test split."
)
parser.add_argument(
    "--config", required=True, help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the serialization directory.",
)

parser.add_argument_group("Compute resource management arguments.")
parser.add_argument(
    "--gpu-ids", required=True, nargs="+", type=int, help="List of GPU IDs to use (-1 for CPU)."
)
parser.add_argument(
    "--cpu-workers", type=int, default=0, help="Number of CPU workers to use for data loading."
)
parser.add_argument(
    "--in-memory", action="store_true", help="Whether to load image features in memory."
)
parser.add_argument(
    "--checkpoint-path", required=True, help="Path to load checkpoint and run inference on."
)
parser.add_argument("--output-path", required=True, help="Path to save predictions (as a JSON).")
parser.add_argument(
    "--evalai-submit", action="store_true", help="Whether to submit the predictions to EvalAI."
)


if __name__ == "__main__":
    _A = parser.parse_args()
    _C = Config(_A.config, _A.config_override)

    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    random.seed(_C.RANDOM_SEED)
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(f"cuda:{_A.gpu_ids[0]}" if _A.gpu_ids[0] >= 0 else "cpu")

    vocabulary = Vocabulary.from_files(_C.DATA.VOCABULARY)

    if(_C.DATA.CBS.WORDFORMS):
        vocabulary = add_constraint_words_to_vocabulary(
            vocabulary, wordforms_tsvpath=_C.DATA.CBS.WORDFORMS
        )
    if(_C.DATA.CBS.WORDFORMS_ATTRIBS):
        vocabulary = add_constraint_words_to_vocabulary(
            vocabulary, wordforms_tsvpath=_C.DATA.CBS.WORDFORMS_ATTRIBS
        )

    EvaluationDatasetClass = (
        EvaluationDatasetWithConstraints if _C.MODEL.USE_CBS else EvaluationDataset
    )
    infer_dataset = EvaluationDatasetClass.from_config(
        _C, vocabulary=vocabulary, in_memory=_A.in_memory
    )


    batch_size = _C.OPTIM.BATCH_SIZE // _C.MODEL.BEAM_SIZE
    batch_size = batch_size or 1
    if _C.MODEL.USE_CBS:
        batch_size = batch_size // (2 ** _C.DATA.CBS.MAX_GIVEN_CONSTRAINTS)
        batch_size = batch_size or 1
    
    batch_size = 1

    infer_dataloader = DataLoader(
        infer_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_A.cpu_workers,
        collate_fn=infer_dataset.collate_fn,
    )

    model = UpDownCaptioner.from_config(_C, vocabulary=vocabulary, device=device).to(device)
    model.load_state_dict(torch.load(_A.checkpoint_path)["model"])

    if len(_A.gpu_ids) > 1 and -1 not in _A.gpu_ids:
        model = nn.DataParallel(model, _A.gpu_ids)


    model.eval()

    predictions = []
    
    counter = 0
    for batch in tqdm(infer_dataloader):

        batch_image_attributes = batch.pop("image_attributes")
        batch_constraint2states = batch.pop("constraint2states")
        batch_candidates = batch.pop("candidates")
        batch_obj_atts = batch.pop("obj_atts")
        
        try:
            fsm = batch.pop("fsm")
        except:
            fsm = None
        
        
        if(fsm):
            for k in range(len(fsm)):
                fsm[k] = fsm[k].to(device)
    
        batch = {key: value.to(device) for key, value in batch.items()}
        
        counter += 1
        print(batch_obj_atts)        
        for k in range(_C.MODEL.N_Z_SAMPLES):        
            with torch.no_grad():

                k_temp = k
                if(not _C.MODEL.STATE_MACHINE_PER_Z_SAMPLE):
                    k = 0
                
                image_features = batch["image_features"]
                num_constraints = batch.get("num_constraints", None)
                sentiment = batch["sentiment"]
                
                try:
                    result = model(
                        image_features,
                        batch_obj_atts,
                        batch_image_attributes,
                        fsm=fsm[k],
                        num_constraints=torch.tensor([num_constraints[k]]).long(),
                        constraints=batch_candidates[k],
                        constraint2states=batch_constraint2states[k],
                        sentiment=sentiment,
                    )
                except:
                    try:                        
                        result = model(
                            image_features,
                            batch_obj_atts,
                            batch_image_attributes,
                            sentiment=sentiment,
                        )
                    except:
                        print("prediction failed.")
                        continue
                
                k = k_temp
                
                batch_predictions = result["predictions"]

            
            for i, image_id in enumerate(batch["image_id"]):
                instance_predictions = batch_predictions[i, :]

                caption = [vocabulary.get_token_from_index(p.item()) for p in instance_predictions]
                eos_occurences = [j for j in range(len(caption)) if caption[j] == "@@BOUNDARY@@"]
                caption = caption[: eos_occurences[0]] if len(eos_occurences) > 0 else caption

                try:
                    predictions.append({"image_id": image_id.item(), "caption": " ".join(caption), "candidates": batch_candidates[i]})
                except:
                    predictions.append({"image_id": image_id.item(), "caption": " ".join(caption)})
                
                print({"image_id": image_id.item(), "caption": " ".join(caption)})

    json.dump(predictions, open(_A.output_path, "w", encoding='utf-8'))
