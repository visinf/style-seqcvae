import random

import numpy as np
import torch
from torch.utils.data import Dataset
from allennlp.data import Vocabulary

from updown.config import Config
from updown.data.readers import CocoCaptionsReader, ConstraintBoxesReader, ImageFeaturesReader, CocoAttributesReader, SenticapReader, ExpertReader
from updown.types import (
    TrainingInstance,
    TrainingBatch,
    EvaluationInstance,
    EvaluationBatch,
)
from updown.utils.constraints import ConstraintFilter, FiniteStateMachineBuilder

class TrainingDataset(Dataset):
    def __init__(
        self,
        vocabulary: Vocabulary,
        captions_jsonpath: str,
        senticap_jsonpath: str,
        expert_jsonpath: str,
        do_load_coco: bool,
        do_load_senticap: bool,
        image_features_h5path: str,
        image_features_h5path_valid: str,
        attribs_dir_path: str,
        use_obj_att_preds,
        att_pred_thresh,
        remove_samples_without_attribs: bool,
        max_caption_length: int = 20,
        in_memory: bool = True,
        sentiment: str = None,
    ):
        self._vocabulary = vocabulary
        self._image_features_reader = ImageFeaturesReader("train", image_features_h5path, in_memory, use_obj_att_preds)
        self._image_features_reader_valid = ImageFeaturesReader("val", image_features_h5path_valid, in_memory, use_obj_att_preds)

        self._captions_reader = None
        if(do_load_coco):
            print("Start loading coco....")
            self._captions_reader = CocoCaptionsReader(captions_jsonpath)
            
            if(do_load_senticap):
                print(len(self))
                print("Start loading senticap....")
                self._senticap_reader = SenticapReader(senticap_jsonpath, train_split=True, val_split=True, sentiment=None)  
                for k in range(20):
                    self._captions_reader._captions.extend(self._senticap_reader._captions)                
                print(len(self))
        
        elif(do_load_senticap):
            print("Start loading senticap....")
            self._captions_reader = SenticapReader(senticap_jsonpath, train_split=True, val_split=True, sentiment=None)
            print(len(self))
 
        self.sentiment = sentiment
        
        if(expert_jsonpath):
            print("Start loading expert captions....")
            self._expert_reader = ExpertReader(expert_jsonpath)
            if(self._captions_reader):
                print("before:", len(self))
                for k in range(1):
                    self._captions_reader._captions.extend(self._expert_reader._captions)
            else:
                self._captions_reader = self._expert_reader
            print("after:",len(self))
            
        if(senticap_jsonpath):
            self._senticap_reader_val = SenticapReader(senticap_jsonpath, test_split=True, sentiment=None)
            n_captions_before = len(self._captions_reader)   
            self._captions_reader._captions = [x for x in self._captions_reader if x[0] not in self._senticap_reader_val._image_ids]
            print((n_captions_before - len(self._captions_reader)), "captions removed from train set as they are part of senticap test set.")

        self.coco_attributes_reader = None
        if(attribs_dir_path):
            self.coco_attributes_reader = CocoAttributesReader(attribs_dir_path)
            image_ids_with_atts = set(self.coco_attributes_reader.image_ids)
            if(remove_samples_without_attribs):
                self._captions_reader._captions = [x for x in self._captions_reader if x[0] in image_ids_with_atts]

        self.use_obj_att_preds = use_obj_att_preds
        self.att_pred_thresh = att_pred_thresh

        if(attribs_dir_path):
            n_captions_before = len(self._captions_reader)   
            self._captions_reader._captions = [x for x in self._captions_reader if x[0] in self._image_features_reader._map 
                                               or x[0] in self._image_features_reader_valid._map]
            print((n_captions_before - len(self._captions_reader)), "captions removed due to missing image features.")

        
        self._max_caption_length = max_caption_length


    @classmethod
    def from_config(cls, config: Config, **kwargs):
        r"""Instantiate this class directly from a :class:`~updown.config.Config`."""
        _C = config
        vocabulary = kwargs.pop("vocabulary")
        return cls(
            vocabulary=vocabulary,
            image_features_h5path=_C.DATA.TRAIN_FEATURES,
            image_features_h5path_valid=_C.DATA.INFER_FEATURES,
            captions_jsonpath=_C.DATA.TRAIN_CAPTIONS,
            senticap_jsonpath=_C.DATA.SENTICAP_CAPTIONS,
            expert_jsonpath = _C.DATA.EXPERT_CAPTIONS,
            do_load_coco=_C.DATA.DO_LOAD_COCO,
            do_load_senticap=_C.DATA.DO_LOAD_SENTICAP,
            attribs_dir_path=_C.DATA.COCO_ATTRIBS_OBJS,
            use_obj_att_preds = _C.DATA.USE_OBJ_ATT_PREDS,
            att_pred_thresh = _C.DATA.ATT_PRED_THRESH,
            remove_samples_without_attribs = _C.DATA.REMOVE_SAMPLES_WITHOUT_ATTRIBS,
            max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
            in_memory=kwargs.pop("in_memory"),
            sentiment=_C.DATA.SENTICAP_SENTIMENT,
        )

    def __len__(self):
        # Number of training examples are number of captions, not number of images.
        return len(self._captions_reader)

    def __getitem__(self, index: int):
        try:
            image_id, caption, sentiment = self._captions_reader[index]
            if(sentiment == 0):
                sentiment = -1
        except:
            image_id, caption = self._captions_reader[index]
            sentiment = 0
            
        try:
            image_features, obj_atts = self._image_features_reader[image_id]
        except:
            image_features, obj_atts = self._image_features_reader_valid[image_id]

        if(self.coco_attributes_reader is not None and obj_atts is not None):
                obj_atts = obj_atts_id2string(self._image_features_reader.obj_id2name, 
                                              self.coco_attributes_reader.attrib_names, 
                                              obj_atts,
                                              self.coco_attributes_reader.avail_obj_names,
                                              self.att_pred_thresh,
                                              self.use_obj_att_preds)

            
        try:
            image_attributes = self.coco_attributes_reader[image_id]
        except:
            image_attributes = None

        # Tokenize caption.
        caption_tokens = [self._vocabulary.get_token_index(c) for c in caption]

        # Pad upto max_caption_length.
        caption_tokens = caption_tokens[: self._max_caption_length]
        caption_tokens.extend(
            [self._vocabulary.get_token_index("@@UNKNOWN@@")]
            * (self._max_caption_length - len(caption_tokens))
        )
        
        item: TrainingInstance = {
            "image_id": image_id,
            "image_features": image_features,
            "caption_tokens": caption_tokens,
            "image_attributes": image_attributes,
            "sentiment": sentiment,
            "obj_atts": obj_atts
        }
        return item

    def collate_fn(self, batch_list):
        # Convert lists of ``image_id``s and ``caption_tokens``s as tensors.
        image_id = torch.tensor([instance["image_id"] for instance in batch_list]).long()
        caption_tokens = torch.tensor(
            [instance["caption_tokens"] for instance in batch_list]
        ).long()
        
        image_attributes = [instance["image_attributes"] for instance in batch_list]
        obj_atts = None
        if(self.coco_attributes_reader is not None and batch_list[0]["obj_atts"] is not None):
            obj_atts = [instance["obj_atts"] for instance in batch_list]


        # Pad adaptive image features in the batch.
        image_features = torch.from_numpy(
            _collate_image_features([instance["image_features"] for instance in batch_list])
        )
        
        sentiment = torch.tensor([instance["sentiment"] for instance in batch_list]).float()
        sentiment = sentiment[:, None]

        batch: TrainingBatch = {
            "image_id": image_id,
            "image_features": image_features,
            "caption_tokens": caption_tokens,
            "image_attributes": image_attributes,
            "sentiment": sentiment,
            "obj_atts": obj_atts,
        }
        return batch


class EvaluationDataset(Dataset):
    def __init__(self, train_image_features_h5path: str,
                 val_image_features_h5path: str,
                 val_captions_jsonpath: str,
                 do_load_coco: bool,
                 do_load_senticap: bool,
                 senticap_jsonpath: str,
                 attribs_dir_path: str,
                 use_obj_att_preds,
                 att_pred_thresh,
                 remove_samples_without_attribs: bool,
                 in_memory: bool = True,
                 sentiment: str = None,
    ):
        self._train_image_features_reader = ImageFeaturesReader("train", train_image_features_h5path, in_memory, use_obj_att_preds)
        self._val_image_features_reader = ImageFeaturesReader("val", val_image_features_h5path, in_memory, use_obj_att_preds)
        
        if(do_load_senticap):
            self._image_ids = sorted(list(self._train_image_features_reader._map.keys()) 
                            + list(self._val_image_features_reader._map.keys()))
        else:
            self._image_ids = sorted(list(self._val_image_features_reader._map.keys()))
        
        
        if(do_load_coco):
            print("Start loading coco....")
            self._captions_reader = CocoCaptionsReader(val_captions_jsonpath)
            
            self._image_ids = list(set([x[0] for x in self._captions_reader]))
        
        if(do_load_senticap):
            print("ATTENTION: SENTICAP EVALUATION!")
            print(len(self))
            print("Start loading senticap....")
            self._senticap_reader = SenticapReader(senticap_jsonpath, test_split=True, sentiment=sentiment)
        
            #self._image_ids = [x for x in self._image_ids if x in self._senticap_reader._image_ids]
            self._image_ids = list(self._senticap_reader._image_ids)
            print(len(self))

        if(attribs_dir_path):
            self.coco_attributes_reader = CocoAttributesReader(attribs_dir_path)
            image_ids_with_atts = set(self.coco_attributes_reader.image_ids)  

            if(remove_samples_without_attribs):
                self._image_ids = [x for x in self._image_ids if x in image_ids_with_atts]
            
            self.coco_attributes_reader.obj_id2obj_name[64] = "plant"
            self.coco_attributes_reader.obj_id2obj_name[72] = "television"
            self.coco_attributes_reader.obj_id2obj_name[76] = "computer keyboard"
            self.coco_attributes_reader.obj_id2obj_name[77] = "mobile phone"
            self.coco_attributes_reader.obj_id2obj_name[78] = "microwave oven"
            self.coco_attributes_reader.obj_id2obj_name[35] = "ski"
            self.coco_attributes_reader.obj_id2obj_name[47] = "mug"
            self.coco_attributes_reader.obj_id2obj_name[60] = "doughnut"
            self.coco_attributes_reader.obj_id2obj_name[75] = "remote control"
            self.coco_attributes_reader.obj_id2obj_name[34] = "flying disc"
            self.coco_attributes_reader.obj_id2obj_name[21] = "cattle"
            self.coco_attributes_reader.obj_id2obj_name[89] = "hair dryer"
        else:
            self.coco_attributes_reader = None
        print(len(self))
        
        self.use_obj_att_preds = use_obj_att_preds
        self.att_pred_thresh = att_pred_thresh

        
        n_images_before = len(self._image_ids)
        self._image_ids = [x for x in self._image_ids if x in self._train_image_features_reader._map 
                                                   or x in self._val_image_features_reader._map]
        print((n_images_before - len(self._image_ids)), "images removed due to missing image features.")
        
        
        self.sentiment = sentiment
        
    @classmethod
    def from_config(cls, config: Config, **kwargs):
        _C = config
        return cls(
            train_image_features_h5path=_C.DATA.TRAIN_FEATURES, 
            val_image_features_h5path=_C.DATA.INFER_FEATURES, 
            val_captions_jsonpath = _C.DATA.INFER_CAPTIONS,
            do_load_coco = _C.DATA.DO_LOAD_COCO,
            attribs_dir_path=_C.DATA.COCO_ATTRIBS_OBJS,
            use_obj_att_preds = _C.DATA.USE_OBJ_ATT_PREDS,
            att_pred_thresh = _C.DATA.ATT_PRED_THRESH,
            remove_samples_without_attribs = _C.DATA.REMOVE_SAMPLES_WITHOUT_ATTRIBS,
            do_load_senticap=_C.DATA.DO_LOAD_SENTICAP,
            senticap_jsonpath=_C.DATA.SENTICAP_CAPTIONS,
            in_memory=kwargs.pop("in_memory"),
            sentiment=_C.DATA.SENTICAP_SENTIMENT,
        )

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, index: int):
        image_id = self._image_ids[index]
        
        try:
            image_features, obj_atts = self._train_image_features_reader[image_id]
        except:
            image_features, obj_atts = self._val_image_features_reader[image_id]

        if(self.coco_attributes_reader is not None and obj_atts is not None):
            obj_atts = obj_atts_id2string(self._train_image_features_reader.obj_id2name, 
                                          self.coco_attributes_reader.attrib_names, 
                                          obj_atts,
                                          self.coco_attributes_reader.avail_obj_names,
                                          self.att_pred_thresh,
                                          self.use_obj_att_preds)
            
        try:
            image_attributes = self.coco_attributes_reader[image_id]
        except:
            image_attributes = None
            
            
        if(self.sentiment == "pos"):
            sentiment = 1
        elif(self.sentiment == "neg"):
            sentiment = -1
        else:
            sentiment = 0
            
            
        item: EvaluationInstance = {"image_id": image_id, 
                                    "image_features": image_features, 
                                    "image_attributes": image_attributes, 
                                    "sentiment": sentiment,
                                    "obj_atts": obj_atts
                                    }
        return item

    def collate_fn(self, batch_list):
        # Convert lists of ``image_id``s and ``caption_tokens``s as tensors.
        image_id = torch.tensor([instance["image_id"] for instance in batch_list]).long()

        image_attributes = [instance["image_attributes"] for instance in batch_list]
        obj_atts = None
        if(self.coco_attributes_reader is not None and batch_list[0]["obj_atts"] is not None):
            obj_atts = [instance["obj_atts"] for instance in batch_list]

        # Pad adaptive image features in the batch.
        image_features = torch.from_numpy(
            _collate_image_features([instance["image_features"] for instance in batch_list])
        )
        
        sentiment = torch.tensor([instance["sentiment"] for instance in batch_list]).float()
        sentiment = sentiment[:, None]     

        batch: EvaluationBatch = {
            "image_id": image_id, 
            "image_features": image_features, 
            "image_attributes": image_attributes, 
            "candidates": None, 
            "constraint2states": None,
            "sentiment": sentiment,
            "obj_atts": obj_atts,
        }
        return batch


class EvaluationDatasetWithConstraints(EvaluationDataset):
    def __init__(
        self,
        vocabulary: Vocabulary,
        train_image_features_h5path: str,
        val_image_features_h5path: str,
        val_captions_jsonpath: str,
        do_load_coco: bool,
        do_load_senticap: bool,
        senticap_jsonpath: str,
        attribs_dir_path: str,
        use_obj_att_preds,
        att_pred_thresh,
        remove_samples_without_attribs: bool,
        boxes_jsonpath: str,
        wordforms_tsvpath: str,
        wordforms_attribs_tsvpath: str,
        hierarchy_jsonpath: str,
        nms_threshold: float = 0.85,
        max_given_objects: int = 2,
        max_given_constraints: int = 3,
        max_words_per_constraint: int = 3,
        state_machine_per_z_sample = False,
        n_z_samples = 1,
        in_memory: bool = True,
        cbs_simple: bool = True,
        sentiment: str = None,
    ):
        super().__init__(train_image_features_h5path, val_image_features_h5path, val_captions_jsonpath, do_load_coco, do_load_senticap, senticap_jsonpath, attribs_dir_path, 
                         use_obj_att_preds, att_pred_thresh, remove_samples_without_attribs, in_memory=in_memory, sentiment=sentiment)

        self._vocabulary = vocabulary
        self._pad_index = vocabulary.get_token_index("@@UNKNOWN@@")
            
        self._max_given_objects = max_given_objects
        self._max_given_constraints = max_given_constraints    
    
        self._boxes_reader = ConstraintBoxesReader(boxes_jsonpath)
        
        self.state_machine_per_z_sample = state_machine_per_z_sample
        self.n_z_samples = n_z_samples

        self._constraint_filter = ConstraintFilter(
            hierarchy_jsonpath, nms_threshold, max_given_objects
        )
        
        
        if(self.coco_attributes_reader):
            self._boxes_reader._class_names[64] = "plant"
            self._boxes_reader._class_names[72] = "television"
            self._boxes_reader._class_names[76] = "computer keyboard"
            self._boxes_reader._class_names[77] = "mobile phone"
            self._boxes_reader._class_names[78] = "microwave oven"
            self._boxes_reader._class_names[35] = "ski"
            self._boxes_reader._class_names[47] = "mug"
            self._boxes_reader._class_names[60] = "doughnut"
            self._boxes_reader._class_names[75] = "remote control"
            self._boxes_reader._class_names[34] = "flying disc"
            self._boxes_reader._class_names[21] = "cattle"
            self._boxes_reader._class_names[89] = "hair dryer"
            
            
            
        self._fsm_builder = FiniteStateMachineBuilder(vocabulary, wordforms_tsvpath, wordforms_attribs_tsvpath, 
                                                      max_given_constraints, max_words_per_constraint, 
                                                      self.coco_attributes_reader is not None)
        
        self.cbs_simple = cbs_simple



    @classmethod
    def from_config(cls, config: Config, **kwargs):
        r"""Instantiate this class directly from a :class:`~updown.config.Config`."""
        _C = config
        vocabulary = kwargs.pop("vocabulary")
        return cls(
            vocabulary=vocabulary,
            train_image_features_h5path=_C.DATA.TRAIN_FEATURES,
            val_image_features_h5path=_C.DATA.INFER_FEATURES,
            val_captions_jsonpath = _C.DATA.INFER_CAPTIONS,
            do_load_coco = _C.DATA.DO_LOAD_COCO,
            do_load_senticap=_C.DATA.DO_LOAD_SENTICAP,
            senticap_jsonpath=_C.DATA.SENTICAP_CAPTIONS,
            attribs_dir_path=_C.DATA.COCO_ATTRIBS_OBJS,
            use_obj_att_preds = _C.DATA.USE_OBJ_ATT_PREDS,
            att_pred_thresh = _C.DATA.ATT_PRED_THRESH,
            remove_samples_without_attribs = _C.DATA.REMOVE_SAMPLES_WITHOUT_ATTRIBS,
            boxes_jsonpath=_C.DATA.CBS.INFER_BOXES,
            wordforms_tsvpath=_C.DATA.CBS.WORDFORMS,
            wordforms_attribs_tsvpath=_C.DATA.CBS.WORDFORMS_ATTRIBS,
            hierarchy_jsonpath=_C.DATA.CBS.CLASS_HIERARCHY,
            max_given_objects=_C.DATA.CBS.MAX_GIVEN_OBJECTS,
            max_given_constraints=_C.DATA.CBS.MAX_GIVEN_CONSTRAINTS,
            max_words_per_constraint=_C.DATA.CBS.MAX_WORDS_PER_CONSTRAINT,
            state_machine_per_z_sample = _C.MODEL.STATE_MACHINE_PER_Z_SAMPLE,
            n_z_samples = _C.MODEL.N_Z_SAMPLES,
            in_memory=kwargs.pop("in_memory"),
            cbs_simple=_C.MODEL.CBS_SIMPLE,
            sentiment=_C.DATA.SENTICAP_SENTIMENT,
        )
        
    def generate_cbs_state_machine(self, item):
    
        # Apply constraint filtering to object class names.
        constraint_boxes = self._boxes_reader[item["image_id"]]

        candidates_obj = dict()
        
        for bbox, classname in zip(constraint_boxes["boxes"], constraint_boxes["class_names"]):
            area = bbox[2]*bbox[3]
            
            if(classname in candidates_obj):
                candidates_obj[classname] += area
            else:
                candidates_obj[classname] = area

        candidates_obj = [k for k, v in sorted(candidates_obj.items(), key=lambda item: item[1], reverse=True)]
        image_attributes = item["image_attributes"]
        
        if(image_attributes):
            image_attributes.sort(key=lambda x:len(x[1]))
        
        if(self.cbs_simple):
            candidates = []
# =============================================================================
#             if(not self.sentiment and not self.coco_attributes_reader):
#                 raise ValueError("sentiment param has to be set either to 'pos' or 'neg' or coco_attributes must be used.")
# =============================================================================
            
            if(self.coco_attributes_reader):
                obj_atts_det = item["obj_atts"]
                objs = {o[0]: set() for o in obj_atts_det}
                atts_all = set()
                candidates = set()
                for obj in obj_atts_det:
                    if(len(obj[1])):
                        for a in obj[1]:
                            att_cleaned = a.split(" ")[-1]
                            if not att_cleaned:
                                att_cleaned = a.split(" ")[-2] 
                            objs[obj[0]].add(att_cleaned)
                            atts_all.add(att_cleaned)
                
                candidates = []

                for obj, atts in objs.items():
                    if(atts):
                        candidates.append(random.sample(atts, 1)[0])
                        
                candidates = candidates[:self._max_given_constraints]

                #candidates = ["all"] * self._max_given_constraints
            else:                
                for k in range(self._max_given_constraints):
                    candidates.append(self.sentiment)
            
            fsm_input = candidates
            #raise SystemExit()
        else:
            candidates = []
            objects_with_attributes = [o[0] for o in image_attributes]
            objects_without_attributes = [[o, []] for o in candidates_obj if o not in objects_with_attributes]
            
            n_atts_per_obj = 2 if len(objects_with_attributes) <= 2 else 1
            #n_atts_per_obj = 2 if len(objects_with_attributes) <= 2 else 1
    
            attribs_already_added = []
            n_objects_added = 0
            
            for o in image_attributes[:3]:
                #print(o[1])
    
                new_cand_attribs = []
                
                atts = [a[0] for a in o[1]]
                atts_prob = [a[1] for a in o[1]]
                
                k = 0 
                while k < n_atts_per_obj:
                    if len(atts) == 0:
                        break
    
                    #sample = random.choices(atts, weights=atts_prob, k=1)[0]
                    sample = random.choices(atts)[0]
                    sample_cleaned = sample.split(" ")[-1]
                    if not sample_cleaned:
                        sample_cleaned = sample.split(" ")[-2]
                    
                    del atts_prob[atts.index(sample)]
                    del atts[atts.index(sample)]
                    
                    if sample not in attribs_already_added:
                        new_cand_attribs.append(sample_cleaned)
                        attribs_already_added.append(sample)
                        
                        k += 1
                candidates.append([o[0], new_cand_attribs])
                n_objects_added += 1
                
            n_avail_slots = min(self._max_given_objects - n_objects_added, 
                                self._max_given_constraints - n_objects_added - len(attribs_already_added))
            candidates.extend(objects_without_attributes[:n_avail_slots])
    
    
            fsm_input = []
            for o in candidates:
                #fsm_input.append(o[0])
                fsm_input.extend(o[1])
            fsm_input = fsm_input[:self._max_given_constraints]
        fsm, nstates, constraint2states = self._fsm_builder.build(fsm_input)
        #print(fsm_input)
        return fsm, nstates, constraint2states, candidates, fsm_input

    def __getitem__(self, index: int):
        item: EvaluationInstance = super().__getitem__(index)
                
        fsm = []
        nstates = []
        num_constraints = []
        candidates = []
        constraint2states = []
        for k in range(self.n_z_samples):
            result = self.generate_cbs_state_machine(item)
            fsm.append(result[0])
            nstates.append(result[1])
            constraint2states.append(result[2])
            candidates.append(result[3])
            num_constraints.append(len(result[4]))
            
            if(not self.state_machine_per_z_sample):
                break


        return {"fsm": fsm, "num_states": nstates, "num_constraints": num_constraints, "candidates": candidates, "constraint2states": constraint2states, **item}

    def collate_fn(
        self, batch_list
    ):

        batch = super().collate_fn(batch_list)

        fsm = []
        for k in range(len(batch_list[0]["fsm"])):
            max_state = batch_list[0]["num_states"][k]
            fsm.append(batch_list[0]["fsm"][k][None, :max_state, :max_state, :])
            
        num_candidates = torch.tensor(batch_list[0]["num_constraints"]).long()
        candidates = batch_list[0]["candidates"]
        constraint2states = batch_list[0]["constraint2states"]
        
        batch.update({"fsm": fsm, "num_constraints": num_candidates, "candidates": candidates, "constraint2states": constraint2states})
        return batch


def _collate_image_features(image_features_list):
    num_boxes = [instance.shape[0] for instance in image_features_list]
    image_feature_size = image_features_list[0].shape[-1]

    image_features = np.zeros(
        (len(image_features_list), max(num_boxes), image_feature_size), dtype=np.float32
    )
    for i, (instance, dim) in enumerate(zip(image_features_list, num_boxes)):
        image_features[i, :dim] = instance
    return image_features


def obj_atts_id2string(obj_voc, atts_voc, obj_atts, avail_obj_names, att_pred_thresh, use_obj_att_preds):
    obj_atts_new = []
    
    for obj in obj_atts:
        obj_new = [obj_voc[obj[0]], []]
        if((not use_obj_att_preds) or (obj_voc[obj[0]] in avail_obj_names)):
            for att in obj[1]:
                if(att[1] >= att_pred_thresh):
                    obj_new[1].append(atts_voc[att[0]])
            
        obj_atts_new.append(obj_new)

    return obj_atts_new