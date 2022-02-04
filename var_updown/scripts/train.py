import argparse
import os
from typing import Any, Dict
import random

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from allennlp.data import Vocabulary

from updown.config import Config
from updown.data.datasets import (
    TrainingDataset
)
from var_updown.models import UpDownCaptioner
from updown.utils.checkpointing import CheckpointManager
from updown.utils.common import cycle
from updown.utils.evalai import NocapsEvaluator
from updown.utils.constraints import add_constraint_words_to_vocabulary


parser = argparse.ArgumentParser("Train an UpDown Captioner on COCO train2017 split.")
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

parser.add_argument_group("Checkpointing related arguments.")
parser.add_argument(
    "--skip-validation",
    action="store_true",
    help="Whether to skip validation and simply serialize checkpoints. This won't track the "
    "best performing checkpoint (obviously). useful for cases where GPU server does not have "
    "internet access and/or checkpoints are validation externally.",
)
parser.add_argument(
    "--serialization-dir",
    default="checkpoints/experiment",
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument(
    "--checkpoint-every",
    default=10000,
    type=int,
    help="Save a checkpoint after every this many epochs/iterations.",
)
parser.add_argument(
    "--start-from-checkpoint",
    default="",
    help="Path to load checkpoint and continue training [only supported for module_training].",
)


if __name__ == "__main__":
    _A = parser.parse_args()
    _C = Config(_A.config, _A.config_override)

    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    os.makedirs(_A.serialization_dir, exist_ok=True)
    _C.dump(os.path.join(_A.serialization_dir, "config.yml"))

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
   
   
    train_dataset = TrainingDataset.from_config(_C, vocabulary=vocabulary, in_memory=_A.in_memory)
    print("final dataset length:", len(train_dataset))
    
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        shuffle=True,
        num_workers=_A.cpu_workers,
        collate_fn=train_dataset.collate_fn,
    )
    
    train_dataloader = cycle(train_dataloader, device)

    model = UpDownCaptioner.from_config(_C, vocabulary=vocabulary, cbs_simple=_C.MODEL.CBS_SIMPLE, device=device).to(device)
    if len(_A.gpu_ids) > 1 and -1 not in _A.gpu_ids:
        model = nn.DataParallel(model, _A.gpu_ids)

    optimizer = optim.SGD(
        model.parameters(),
        lr=_C.OPTIM.LR,
        momentum=_C.OPTIM.MOMENTUM,
        weight_decay=_C.OPTIM.WEIGHT_DECAY,
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(  # type: ignore
        optimizer, lr_lambda=lambda iteration: 1 - iteration / _C.OPTIM.NUM_ITERATIONS
    )

    tensorboard_writer = SummaryWriter(logdir=_A.serialization_dir)

    checkpoint_manager = CheckpointManager(model, optimizer, _A.serialization_dir, mode="max")

    evaluator = NocapsEvaluator(phase="val")

    if _A.start_from_checkpoint != "":
        training_checkpoint: Dict[str, Any] = torch.load(_A.start_from_checkpoint)
        for key in training_checkpoint:
            if key == "optimizer":
                optimizer.load_state_dict(training_checkpoint[key])
            else:
                model.load_state_dict(training_checkpoint[key])
        start_iteration = 1
    else:
        start_iteration = 1


    for iteration in tqdm(range(start_iteration, _C.OPTIM.NUM_ITERATIONS + 1)):

        if(iteration > _C.OPTIM.EPOCH_START_DECODER_TRAINING or iteration % _C.OPTIM.BEFORE_UPDATE_DECODER_EVERY == 0):
            for param in model._updown_cell._language_lstm_cell_decoder.parameters():
                param.requires_grad = True
        else:
            for param in model._updown_cell._language_lstm_cell_decoder.parameters():
                param.requires_grad = False
        
        batch = next(train_dataloader)


        optimizer.zero_grad()
        output_dict = model(batch["image_features"], batch["obj_atts"], batch["image_attributes"], batch["caption_tokens"], batch["sentiment"])
        reconstr_loss = output_dict["loss"].mean()
        kld_loss = output_dict["kld"].mean()
               
        loss = reconstr_loss + kld_loss / _C.MODEL.KLD_WEIGHT        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), _C.OPTIM.CLIP_GRADIENTS)

        optimizer.step()
        lr_scheduler.step()

        # Log loss and learning rate to tensorboard.
        tensorboard_writer.add_scalar("1reconstr_loss", reconstr_loss, iteration)
        tensorboard_writer.add_scalar("2kld_loss", kld_loss, iteration)
        tensorboard_writer.add_scalar("3loss", loss, iteration)
        tensorboard_writer.add_scalar("4learning_rate", optimizer.param_groups[0]["lr"], iteration)
        
        if(iteration % 2000 == 0):
            print("{:6f}    {:6f}    {:6f}".format(loss.item(), reconstr_loss.item(), kld_loss.item()))
        
        if iteration % _A.checkpoint_every == 0:
            checkpoint_manager.step(0.0, iteration)
