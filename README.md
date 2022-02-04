# Diverse Image Captioning with Grounded Style

This repository is the PyTorch implementation of the paper:

**Diverse Image Captioning with Grounded Style** \
Franz Klein, [Shweta Mahajan](https://www.visinf.tu-darmstadt.de/visinf/team_members/smahajan/smahajan.en.jsp), [Stefan Roth](https://www.visinf.tu-darmstadt.de/visinf/team_members/sroth/sroth.en.jsp).
In GCPR 2021.

## Requirements
This codebase is written in Python 3.6 and CUDA 9.0. \
Required Python packages are summarized in ``requirements.txt``.

## Overview
    .
    ├── data                   # Senticap/COCO Attributes wordforms, corresponding synsets and SentiWordNet scores
    ├── eval                   # Evaluation tools based on Senticap and COCO reference captions
    ├── frcnn                  # Faster R-CNN implementation augmented by attribute detection component and image feature extraction funtionality
    ├── misc                   # Different scripts for pre- and postprocessing
    ├── updown_baseline        # Implementation ot BU-TD image captioning + CBS; augmented by readers for Senticap, COCO Attributes
    ├── var_updown             # Implementation of the Style-SeqCVAE model introduced in this work
    ├── requirements.txt
    ├── LICENSE
    └── README.md

## Setup
1. Initially, download and store the following datasets:

| Dataset | Website |
|:-:|:-:|
| Senticap | [Link](http://users.cecs.anu.edu.au/~u4534172/senticap.html) |
| COCO | [Link](https://cocodataset.org/#download) |
| COCO Attributes | [Link](http://cs.brown.edu/~gmpatter/cocottributes.html) |
| SentiWordNet (optionally)| [Link](https://github.com/aesuli/SentiWordNet) |

2. Faster R-CNN preparation \
Please follow the instructions of the original [implementation](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) to setup this modified Faster R-CNN codebase.
3. Style-SeqCVAE preparation \
Please follow the instructions of the original [implementation](https://nocaps.org/updown-baseline/setup_dependencies.html) for setup.
4. Additional steps:
    - Create a COCO/Senticap vocabulary by running the following command: 
      ```
      python scripts/build_vocabulary.py -c /path/to/coco/captions.json 
                                         -o /path/to/vocabulary/target/ 
                                         -s /path/to/senticap/captions.json
      ```
    - Preprocess the Coco Attributes dataset by running `misc/gen_coco_attribute_objs.py`
    - Augment COCO captions by COCO Attributes with `misc/prep_coco_att_data.py` or Senticap adjectives with `misc/prep_senti_data.py`
    - The SentiGloVe latent space can by prepared by running `misc/prep_expl_lat_space.py`

## Training and Evaluation

### Faster R-CNN training and extraction of image features/attribute detections
The original implementation is augmented by an attribute detection layer and can be trained using COCO + COCO Attributes \
Add `--cocoatts` as runtime parameter to activate attribute detection.
```
CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --dataset coco --net res101 \
                   --bs 16 --nw 2 \
                   --lr 0.01 --lr_decay_step 4 \
                   --cuda --cocoatts
```
Adding `--senticap` as runtime parameter ensures that training ignores images that occur in the Senticap test split.

To extract image features and corresponding attribute detections run the modified test script with `--feat_extract` as parameter. 
```
python test_net.py --dataset coco --net res101 \
                   --checksession 1 --checkepoch 10 --checkpoint 14657 \
                   --cuda --feat_extract
```

### Style-Sequential CVAE training/evaluation
To start training run `var_updown/scripts/train.py` similar to e.g.
```
python scripts/train.py --config configs/updown_plus_cbs_nocaps_val.yaml \
                        --serialization-dir /path/to/checkpoint/destination/folder \
                        --gpu-ids 0
```

For evaluation of a trained model first run the following command to store predictions in a JSON file.
```
python scripts/inference.py --config configs/updown_plus_cbs_nocaps_val.yaml \ 
                            --checkpoint-path /path/to/checkpoint.pth \
                            --output-path /path/to/output.json \
                            --gpu-ids 0
```

To evaluate generated captions based on COCO or Senticap ground truth captions set the paths and config parameters in `eval/eval.py` and run it.
The following metrics are available:  
- BLEU  
- METEOR  
- ROUGE
- CIDER
- n-gram diversity
- sentiment accuracy, sentiment recall
- Top-* oracle score calculation for each metric if multiple candidate captions given per image_id

## Acknowledgements
This project is based on the following publicly available repositories
- https://github.com/nocaps-org/updown-baseline
- https://github.com/jwyang/faster-rcnn.pytorch

We would like to thank all who have contributed to them.


## Citations

If you use our code, please cite our GCPR 2021 paper:

    @inproceedings{Klein:2021:DICWGS,
        title = {Diverse Image Captioning with Grounded Style},
        author = {Franz Klein and Shweta Mahajan and Stefan Roth},
        booktitle = {Pattern Recognition, 43rd DAGM German Conference, DAGM GCPR 2021},
        year = {2021}}
