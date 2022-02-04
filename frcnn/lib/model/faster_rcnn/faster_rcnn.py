import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, n_classes_att=0, att_counts=None):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_classes_att = n_classes_att
        
        self.att_counts = att_counts
        
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

        #self._embedding_layer = nn.Embedding(self.n_classes, 256)
        

    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_atts=None):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
    
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        
        if(gt_atts is not None):
            gt_atts = gt_atts.data
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes, gt_atts)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, rois_label_atts = roi_data

        if self.training:
        
# =============================================================================
#             print("rois_label.shape:", rois_label.shape)
#             print("rois_label_atts.shape:", rois_label_atts.shape)
#             
#             for batch in range(rois_label.shape[0]):
#                 for bbox in range(rois_label.shape[1]):
#                     print(rois_label[batch][bbox].item(), "    ", rois[batch][bbox].detach().cpu().numpy())
#                     if(rois_label_atts[batch][bbox].sum() > 0):
#                         print("atts:", torch.squeeze(rois_label_atts[batch][bbox].nonzero()).detach().cpu().numpy())
#                         print("-----")
# =============================================================================
        
        
# =============================================================================
#             for b in range (rois.shape[0]):
#                 for bx in range (rois.shape[1]):
#                     print(rois[b][bx], "       ", rois_label[b][bx])
#                 print("-----------------------------------------")
# =============================================================================
            
            
            #print("rois:", rois)
            #print("rois_label:", rois_label)
            #print("rois_target:", rois_target.sum(axis=2))
# =============================================================================
#             print("rois_target.shape:", rois_target.shape)
#             print("rois.shape:", rois.shape, "    rois_label.shape:", rois_label.shape)
#             print("rois_label_atts:", rois_label_atts.shape)
# =============================================================================


            rois_label = Variable(rois_label.view(-1).long())
            rois_label_atts = Variable(rois_label_atts.view(-1, self.n_classes_att))
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            if(gt_atts is not None):
                rois_label_atts = Variable(rois_label_atts.view(-1, self.n_classes_att))
                rois_label = Variable(rois_label.view(-1).long())
            else:
                rois_label_atts = None
                rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        
#        print(cls_prob.shape)

        #test = cls_prob.max(axis=1)
        #test = np.argwhere(cls_prob.cpu() > 0.85)
#        print(test)
        #print(test.shape)
#        print(cls_prob.cpu()[test])
        
        #print(rois_label)

# =============================================================================
#         if(self.training):
#             cls_emb = self._embedding_layer(rois_label)
#         else:
#             cls_emb = self._embedding_layer(cls_prob.argmax(axis=1))
#             
#         pooled_feat = torch.cat((pooled_feat, cls_emb), 1)
# =============================================================================
        
        if(self.n_classes_att):
            cls_score_atts = self.RCNN_cls_score_atts(pooled_feat)
            cls_prob_atts = torch.sigmoid(cls_score_atts)
                
        RCNN_loss_cls = 0
        RCNN_loss_cls_atts = torch.zeros(1).mean().cuda()
        RCNN_loss_bbox = 0
        
# =============================================================================
#         print("cls_score.shape:", cls_score.shape)        
#         print("rois_label.shape:", rois_label.shape)
# 
#         print("cls_score_atts.shape:", cls_score_atts.shape)        
#         print("rois_label_atts.shape:",rois_label_atts.shape)
# =============================================================================
        

        if self.training:
            # classification loss
            #print(rois_label)
# =============================================================================
#             print("rois_label.shape:", rois_label.shape)
#             print("................................")
# =============================================================================
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            
            if(self.n_classes_att and rois_label_atts.sum()):
                #RCNN_loss_cls_atts = F.binary_cross_entropy(cls_prob_atts, rois_label_atts, reduction="none")
                #valid_indizes = rois_label_atts.sum(axis=1).nonzero()
                #RCNN_loss_cls_atts = RCNN_loss_cls_atts[valid_indizes].mean()
            
                RCNN_loss_cls_atts = self.CB_loss(rois_label_atts, cls_score_atts, self.att_counts, len(self.att_counts), "sigmoid", 0.9, 1) #0.9999, 2
                
                valid_indizes = rois_label_atts.sum(axis=1).nonzero()
                RCNN_loss_cls_atts = RCNN_loss_cls_atts[valid_indizes].mean()

# =============================================================================
#                 RCNN_loss_cls_atts = - F.binary_cross_entropy(cls_prob_atts, rois_label_atts, reduction="none")
#                 pt = torch.exp(RCNN_loss_cls_atts)
#                 
#                 focal_loss = -( (1-pt)**2 ) * RCNN_loss_cls_atts
#                 balanced_focal_loss = 0.25 * focal_loss
#                 
#                 valid_indizes = rois_label_atts.sum(axis=1).nonzero()
#                 RCNN_loss_cls_atts = balanced_focal_loss[valid_indizes].mean()
# =============================================================================

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        cls_prob_atts = cls_prob_atts.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        
        return rois, cls_prob, cls_prob_atts, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_cls_atts, RCNN_loss_bbox, rois_label, pooled_feat, rois_label_atts


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_atts, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.RCNN_cls_score_atts.bias.data = -torch.log((self.n_classes_att-1) * torch.ones(self.RCNN_cls_score_atts.bias.shape))
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
        
        
    def focal_loss(self, labels, logits, alpha, gamma):
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
          labels: A float tensor of size [batch, num_classes].
          logits: A float tensor of size [batch, num_classes].
          alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
          gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
          focal_loss: A float32 scalar representing normalized total loss.
        """    
        BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")
    
        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
                torch.exp(-1.0 * logits)))
    
        loss = modulator * BCLoss
    
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    
        focal_loss /= torch.sum(labels)
        return focal_loss
    
    
    
    def CB_loss(self, labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
    
        labels_one_hot = labels
    
        weights = torch.tensor(weights).float().cuda()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,no_of_classes)
    
        if loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights, gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights, reduction="none")
        elif loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy_with_logits(input = pred, target = labels_one_hot, weight = weights, reduction="none")
        return cb_loss
