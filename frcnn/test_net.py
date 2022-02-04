# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import json

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import h5py


try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--senticap', dest='senticap',
                      help='whether senticap test split images should be removed from train set',
                      action='store_true')
  parser.add_argument('--cocoatts', dest='cocoatts',
                      help='whether cocoatts should be detected',
                      action='store_true')
  parser.add_argument('--feat_extract', dest='feat_extract',
                      help='feature extraction instead of normal test mode',
                      action='store_true')

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      #args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      #args.imdbval_name = "coco_2014_minival"
      args.imdb_name = "coco_2014_train"
      args.imdbval_name = "coco_2014_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)
    
  cfg.SENTICAP = args.senticap
  cfg.COCOATTS = args.cocoatts
  FEAT_EXTRACT = args.feat_extract
  
  if(FEAT_EXTRACT):
      start_index = 0
      thresh = 0.5
  else:
      start_index = 1
      thresh = 0.05

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name, training=False, 
                                                        senticap=cfg.SENTICAP, cocoatts=cfg.COCOATTS)


    
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic, n_classes_att=imdb._n_atts)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05

  save_name = 'faster_rcnn_10'
  num_images = len(roidb)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]
  all_pooled_feats = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]
  all_rois_labels = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]
  all_atts = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]
  all_atts_det = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]


  if(FEAT_EXTRACT):
      split = True
  else:
      split = False

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, imdb._n_atts, training=split, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  gt_atts = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    gt_atts = gt_atts.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)
  gt_atts = Variable(gt_atts)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')
  att_det_file = os.path.join(output_dir, 'detections_att.json')

  counter = 0

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  
  if(FEAT_EXTRACT):
      hf = h5py.File('./features.h5', 'w')
      dt = h5py.vlen_dtype(np.dtype('float32'))
      dt2 = h5py.vlen_dtype(np.dtype('uint8'))
      hf.create_dataset("features", (num_images,), dtype=dt)
      hf.create_dataset("obj_atts", (num_images,), dtype=dt2)
      hf.create_dataset("obj_atts_det", (num_images,), dtype=dt)
      hf.create_dataset("image_id", (num_images,), dtype=int)
      hf.create_dataset("num_boxes", (num_images,), dtype=int)
  
  att_det_output = {}
  
  failed_counter = 0
  
  for i in range(num_images):

      try:
          data = next(data_iter)
      except:
          failed_counter += 1
          print("failed", failed_counter)
          continue
        
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])
              gt_atts.resize_(data[4].size()).copy_(data[4])
              image_id = data[5].item()

      
      att_input = None
      
      if(FEAT_EXTRACT):
          att_input = gt_atts

      det_tic = time.time()
      rois, cls_prob, cls_prob_atts, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_cls_atts, \
      RCNN_loss_bbox, rois_label, pooled_feat, atts = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, att_input)


      scores = cls_prob.data
      scores_atts = cls_prob_atts.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      
      scores_atts = scores_atts.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
      
      att_det_output_obj = {}
      for j in xrange(start_index, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            cls_scores_atts = scores_atts[inds]
            cls_pooled_feat = pooled_feat[inds]
            
            if(FEAT_EXTRACT):
              cls_rois_label = rois_label[inds]
              cls_atts = atts[inds]
            
            
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            
            cls_dets = cls_dets[order]
            cls_scores_atts = cls_scores_atts[order]
            cls_pooled_feat = cls_pooled_feat[order]
            if(FEAT_EXTRACT):
              cls_rois_label = cls_rois_label[order]
              cls_atts = cls_atts[order]

            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            cls_scores_atts = cls_scores_atts[keep.view(-1).long()]
            cls_pooled_feat = cls_pooled_feat[keep.view(-1).long()]
            if(FEAT_EXTRACT):
              cls_rois_label = cls_rois_label[keep.view(-1).long()]
              cls_atts = cls_atts[keep.view(-1).long()]
            
            
            
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.5)
            all_boxes[j][i] = cls_dets.cpu().numpy()
            all_pooled_feats[j][i] = cls_pooled_feat.cpu().detach().numpy()
            if(FEAT_EXTRACT):
              all_rois_labels[j][i] = cls_rois_label.cpu().detach().numpy()
              all_atts[j][i] = cls_atts.cpu().detach().numpy()
              all_atts_det[j][i] = cls_scores_atts.cpu().detach().numpy()
          else:
            all_boxes[j][i] = empty_array
            all_pooled_feats[j][i] = empty_array
            if(FEAT_EXTRACT):
              all_rois_labels[j][i] = empty_array
              all_atts[j][i] = empty_array
              all_atts_det[j][i] = empty_array


      att_det_output[image_id] = att_det_output_obj
      
      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(start_index, imdb.num_classes)])
          
          if(FEAT_EXTRACT):
              image_boxes = []
              pooled_feats = []
              rois_labels = []
              rois_dets = []
              atts = []
              atts_det = []
              for j in xrange(start_index, imdb.num_classes):
                  boxes_per_class = all_boxes[j][i][:, :4]
                  pooled_feat = all_pooled_feats[j][i]
                  rois_label = all_rois_labels[j][i]
                  rois_det = all_boxes[j][i][:, 4]
                  att = all_atts[j][i]
                  att_det = all_atts_det[j][i]
                  if(boxes_per_class.shape[0] > 0):
                      image_boxes.append(boxes_per_class)
                      pooled_feats.append(pooled_feat)
                      rois_labels.append(rois_label)
                      rois_dets.extend([j] * rois_label.shape[0])
                      atts.append(att)
                      atts_det.append(att_det)

              image_boxes = np.concatenate(image_boxes)
              pooled_feats = np.concatenate(pooled_feats)
              rois_labels = np.concatenate(rois_labels)
              rois_dets = np.array(rois_dets)
              atts = np.concatenate(atts)
              atts_det = np.concatenate(atts_det)
          
          #print(image_boxes.shape)
          
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(start_index, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]
                  all_pooled_feats[j][i] = all_pooled_feats[j][i][keep, :]
                  if(FEAT_EXTRACT):
                    all_rois_labels[j][i] = all_rois_labels[j][i][keep]
                    all_atts[j][i] = all_atts[j][i][keep, :]
                    all_atts_det[j][i] = all_atts_det[j][i][keep, :]
            
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(start_index, imdb.num_classes)])
          
          if(FEAT_EXTRACT):
              image_boxes = []
              pooled_feats = []
              rois_labels = []
              rois_dets = []
              atts = []
              atts_det = []
              for j in xrange(start_index, imdb.num_classes):
                  boxes_per_class = all_boxes[j][i][:, :4]
                  pooled_feat = all_pooled_feats[j][i]
                  rois_label = all_rois_labels[j][i]
                  rois_det = all_boxes[j][i][:, 4]
                  att = all_atts[j][i]
                  att_det = all_atts_det[j][i]
                  if(boxes_per_class.shape[0] > 0):
                      image_boxes.append(boxes_per_class)
                      pooled_feats.append(pooled_feat)
                      rois_labels.append(rois_label)
                      rois_dets.extend([j] * rois_label.shape[0])
                      atts.append(att)
                      atts_det.append(att_det)

              image_boxes = np.concatenate(image_boxes)
              pooled_feats = np.concatenate(pooled_feats)
              rois_labels = np.concatenate(rois_labels)
              rois_dets = np.array(rois_dets)
              atts = np.concatenate(atts)
              atts_det = np.concatenate(atts_det)

            
              final = []
              final2 = []
              
              if(rois_labels.sum() > 0):
                  for idx, obj_idx in enumerate(rois_labels):
                      final.append(obj_idx)

                      att_indizes = atts[idx].nonzero()
                      for a_idx in att_indizes[0]:
                          final.append(a_idx + 100)
               

                  for idx, obj_idx in enumerate(rois_dets):
                      final2.append(obj_idx)

                      att_indizes = np.argwhere(atts_det[idx] > 0.3)
                      for a_idx in att_indizes:
                          final2.append(a_idx.item() + 100)
                          final2.append(atts_det[idx][a_idx].item())   
              

              hf["features"][i] = pooled_feats.flatten()
              hf["num_boxes"][i] = pooled_feats.shape[0]
              hf["image_id"][i] = image_id
              hf["obj_atts"][i] = np.asarray(final, dtype=np.uint8)
              hf["obj_atts_det"][i] = np.asarray(final2, dtype=np.float32)
                  
      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()



      if vis:
          path = os.path.join("result_vis" , str(i) + '.jpg')
          cv2.imwrite(path, im2show)

  if(FEAT_EXTRACT):
      hf.close()

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
      
  with open(att_det_file, 'w') as f2:
      json.dump(att_det_output, f2)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
