from yacs.config import CfgNode as CN


class Config(object):
    def __init__(self, config_file = None, config_override = []):
        _C = CN()        

        # logging
        _C.LOG_TO_FILE = True
        _C.CHECKPOINT_EVERY_N_EPOCHS = 10
        _C.PRINT_EVERY_N_BATCHES = 100

        _C.RANDOM_SEED = 0

        _C.DATA = CN()
        _C.DATA.VOCABULARY = "data/vocabulary"

        _C.DATA.TRAIN_FEATURES = "data/coco_train2017_vg_detector_features_adaptive.h5"
        _C.DATA.INFER_FEATURES = "data/nocaps_val_vg_detector_features_adaptive.h5"

        # DATA.INFER_CAPTIONS don't contain the captions, just the image info.
        _C.DATA.TRAIN_CAPTIONS = "data/coco/captions_train2017.json"
        _C.DATA.INFER_CAPTIONS = "data/nocaps/nocaps_val_image_info.json"
        
        _C.DATA.SENTICAP_CAPTIONS = ""
        _C.DATA.DO_LOAD_COCO = True
        _C.DATA.DO_LOAD_SENTICAP = False
        _C.DATA.SENTICAP_SENTIMENT = ""


        _C.DATA.EXPERT_CAPTIONS = ""

        _C.DATA.COCO_ATTRIBS_OBJS = ""
        _C.DATA.REMOVE_SAMPLES_WITHOUT_ATTRIBS = False
        _C.DATA.USE_OBJ_ATT_PREDS = False
        _C.DATA.ATT_PRED_THRESH = 0.3

        _C.DATA.MAX_CAPTION_LENGTH = 20
        
        # There's no parameter as DATA.CBS.TRAIN_BOXES because CBS is inference-only.
        _C.DATA.CBS = CN()
        _C.DATA.CBS.INFER_BOXES = "data/nocaps_val_oi_detector_boxes.json"
        _C.DATA.CBS.CLASS_HIERARCHY = "data/cbs/class_hierarchy.json"
        _C.DATA.CBS.WORDFORMS = ""
        _C.DATA.CBS.WORDFORMS_ATTRIBS = ""

        _C.DATA.CBS.NMS_THRESHOLD = 0.85
        _C.DATA.CBS.MAX_GIVEN_OBJECTS = 2
        _C.DATA.CBS.MAX_GIVEN_CONSTRAINTS = 3
        _C.DATA.CBS.MAX_WORDS_PER_CONSTRAINT = 3

        _C.MODEL = CN()
        _C.MODEL.IMAGE_FEATURE_SIZE = 2048
        _C.MODEL.EMBEDDING_SIZE = 1000
        _C.MODEL.HIDDEN_SIZE = 1200
        _C.MODEL.ATTENTION_PROJECTION_SIZE = 768
        _C.MODEL.BEAM_SIZE = 5
        _C.MODEL.USE_CBS = False
        _C.MODEL.CBS_SIMPLE = True
        _C.MODEL.MIN_CONSTRAINTS_TO_SATISFY = 2
        
        

        # MODEL CONFIGURATION
        _C.MODEL.PRIOR_MODE = "AG" # "LSTM", "CVAE", "GMM", "AG"
        _C.MODEL.DO_USE_CLUSTER_VECTOR = True
        _C.MODEL.FC_LAYER_PER_ATTRIB = True
        _C.MODEL.NUM_LSTM_LAYERS = 1
        _C.MODEL.LSTM_DROPOUT = 0.1 #ONLY HAS EFFECT FOR NUM_LSTM_LAYERS > 1

        _C.MODEL.Z_SPACE = 150
        _C.MODEL.SENTIMENT_VAE = 0
        _C.MODEL.SENTI_PRIOR_MULTIP = 1.0
        _C.MODEL.LATENT_EMBEDDING_MULTIP = 1.0
        _C.MODEL.KLD_WEIGHT = 750
        _C.MODEL.N_Z_SAMPLES = 0
        _C.MODEL.STATE_MACHINE_PER_Z_SAMPLE = False
        _C.MODEL.LATENT_EMBEDDING = "glove"
        
        _C.MODEL.PRIOR_STD = 1.0 # OR 0.1
        _C.MODEL.SIMPLE_VAE = True
    
        _C.MODEL.DO_USE_KLD_ANNEALING = False
        _C.MODEL.KLD_DECREASING = False
        _C.MODEL.KLD_INITIAL_WEIGHT = 2.0 # INITIAL KLD WEIGHT
        _C.MODEL.KLD_ANNEALING_PER_EPOCH = 0.25  # SHOULD BE FRACTION OF 1
        _C.MODEL.KLD_N_EPOCHS_BEFORE_RESET = 4        


        _C.OPTIM = CN()
        _C.OPTIM.BATCH_SIZE = 150
        _C.OPTIM.NUM_ITERATIONS = 70000
        _C.OPTIM.LR = 0.015
        _C.OPTIM.MOMENTUM = 0.9
        
        _C.OPTIM.LR_DECAY_EVERY_N = 7
        _C.OPTIM.LR_DECAY = 0.5
        _C.OPTIM.LR_DECAY_START_EPOCH = 10
        
        _C.OPTIM.WEIGHT_DECAY = 0.001
        _C.OPTIM.CLIP_GRADIENTS = 12.5
        
        _C.OPTIM.EPOCH_START_DECODER_TRAINING = 40000
        _C.OPTIM.BEFORE_UPDATE_DECODER_EVERY = 30

        # Override parameter values from YAML file first, then from override list.
        self._C = _C
        if config_file is not None:
            self._C.merge_from_file(config_file)
        self._C.merge_from_list(config_override)

        # Do any sort of validations required for the config.
        self._validate()

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path):
        r"""
        Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def _validate(self):
        r"""
        Perform all validations to raise error if there are parameters with conflicting values.
        """
        if self._C.MODEL.USE_CBS:
            assert self._C.MODEL.EMBEDDING_SIZE == 300 or self._C.MODEL.EMBEDDING_SIZE == 600, "Word embeddings must be initialized with"
            " fixed GloVe Embeddings (300 dim) for performing CBS decoding during inference. "
            f"Found MODEL.EMBEDDING_SIZE as {self._C.MODEL.EMBEDDING_SIZE} instead."

        assert (
            self._C.MODEL.MIN_CONSTRAINTS_TO_SATISFY <= self._C.DATA.CBS.MAX_GIVEN_CONSTRAINTS
        ), "Satisfying more constraints than maximum specified is not possible."

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __str__(self):
        common_string: str = str(CN({"RANDOM_SEED": self._C.RANDOM_SEED})) + "\n"
        common_string += str(CN({"DATA": self._C.DATA})) + "\n"
        common_string += str(CN({"MODEL": self._C.MODEL})) + "\n"
        common_string += str(CN({"OPTIM": self._C.OPTIM})) + "\n"

        return common_string

    def __repr__(self):
        return self._C.__repr__()
