
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.PIN_MEMORY = True
_C.VERBOSE = True
_C.NUM_JOINTS = 8 # default key point number
# _C.NUM_JOINTS = 13 # fashion keypoint number
_C.IS_TWOSTAGE = False # use twostage
_C.CONFIDENCE = 0.3 # 置信度
_C.SINGLE_CROP = True # 置信度
_C.ORI_HEIGHT = 480 # 原图实际高度H
_C.ORI_WIDTH = 480 # 原图实际宽度W
_C.LABEL_DICT = ([('左',0),('右',1)]) # label



# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.COARSE_BB = CN(new_allowed=True)
_C.MODEL.COARSE_BB.NAME = 'convnextv2_femto'
_C.MODEL.COARSE_BB.PRETRAINED = '/home/rudolfmaxx/syt_kpt_python/pretrained_models/convnextv2_femto_1k_224_fcmae.pt'
_C.MODEL.COARSE_BB.fpn_ch = 128
_C.MODEL.COARSE_HEAD = CN(new_allowed=True)
_C.MODEL.COARSE_HEAD.NAME = 'HMHead'
_C.MODEL.COARSE_HEAD.up_stages = 0
_C.MODEL.COARSE_HEAD.input_channels = [128]
_C.MODEL.COARSE_HEAD.output_channels = 2
_C.MODEL.COARSE_HEAD.upsample_method = 'bilinear'
_C.MODEL.COARSE_HEAD.concat_dims = []
_C.MODEL.COARSE_HEAD.concat_feat_names = []
_C.MODEL.COARSE_HEAD.concat_last_logit = False


_C.MODEL.REFINE_BB = CN(new_allowed=True)
_C.MODEL.REFINE_BB.NAME = 'convnextv2_femto'
_C.MODEL.REFINE_BB.PRETRAINED = '/home/rudolfmaxx/syt_kpt_python/pretrained_models/convnextv2_femto_1k_224_fcmae.pt'
_C.MODEL.REFINE_BB.fpn_ch = 128
_C.MODEL.REFINE_HEAD = CN(new_allowed=True)
_C.MODEL.REFINE_HEAD.NAME = 'HMHead'
_C.MODEL.REFINE_HEAD.up_stages = 1
_C.MODEL.REFINE_HEAD.input_channels = [256, 128]
# _C.MODEL.REFINE_HEAD.output_channels = 1 # 因为二阶段的输入特征图是crop只有一个点
_C.MODEL.REFINE_HEAD.output_channels = 2     # 输入crop图存在多个点时
_C.MODEL.REFINE_HEAD.upsample_method = 'bilinear'
_C.MODEL.REFINE_HEAD.concat_dims = []
_C.MODEL.REFINE_HEAD.concat_feat_names = []
_C.MODEL.REFINE_HEAD.concat_last_logit = True

_C.MODEL.WMASK = False

_C.MODEL.COARSE_IN_SIZE = (384, 288)
# _C.MODEL.COARSE_IN_SIZE = (480, 640)
_C.MODEL.CUT_OUT_SIZE = 192
# _C.MODEL.CUT_OUT_SIZE = (256, 192)
_C.MODEL.CAT_FEAT = True
_C.MODEL.coarse_cat_refine = True


_C.LOSS = CN(new_allowed=True)
_C.LOSS.HM_LOSS_MODE = 'l2' # l2: L2 norm; FL: Focal Loss


# # training data augmentation
# _C.DATASET.MAX_ROTATION = 30
# _C.DATASET.MIN_SCALE = 0.75
# _C.DATASET.MAX_SCALE = 1.25
# _C.DATASET.SCALE_TYPE = 'short'
# _C.DATASET.MAX_TRANSLATE = 40
# _C.DATASET.INPUT_SIZE = 512
# _C.DATASET.OUTPUT_SIZE = [128, 256, 512]
# _C.DATASET.FLIP = 0.5
# _C.DATASET.SLICE_RATIO = 0.5

# # heatmap generator (default is OUTPUT_SIZE/64)
# _C.DATASET.SIGMA = -1
# _C.DATASET.BASE_SIZE = 16
# _C.DATASET.BASE_SIGMA = 2.0
# _C.DATASET.INT_SIGMA = False

# _C.DATASET.WITH_CENTER = False

# train
_C.TRAIN = CN()

# 学习率相关参数
# LR: 初始学习率，推荐值范围 0.001 - 0.0001
# LR_FACTOR: 学习率衰减因子
# LR_STEPS: 学习率衰减的epoch节点
_C.TRAIN.LR_FACTOR = 0.001
_C.TRAIN.LR_STEPS = [90, 110]
_C.TRAIN.LR = 0.001

# 学习率预热参数
_C.TRAIN.WARM_UP = CN()
_C.TRAIN.WARM_UP.IN = True
_C.TRAIN.WARM_UP.START_FACTOR = 0.1
_C.TRAIN.WARM_UP.STEPS = 5

# 优化器相关参数
# OPTIMIZER: 优化器类型 ('sgd', 'adam', 'adamw')
# WD: 权重衰减系数，推荐值范围 0.01 - 0.0001
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.99
_C.TRAIN.WD = 0.001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.EPOCH_START = 0
_C.TRAIN.EPOCH_UNFREEZE = 0
_C.TRAIN.EPOCH_END = 150

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.PRINT_FREQ = 100
_C.TRAIN.REC_PREQ = 3000
_C.TRAIN.VALID_INTERVAL = 5
_C.TRAIN.SAVE_INTERVAL = 5


_C.TRAIN.SHUFFLE = True

# DATASET related params
_C.TRAIN.DATASET = CN()
_C.TRAIN.DATASET.DATASET = 'syt_rag'
_C.TRAIN.DATASET.ROOT = '/home/rudolfmaxx/syt_kpt_python/kpt_data/train'
_C.TRAIN.DATASET.NAME = 'auged'
# _C.TRAIN.DATASET.OUT_RES_COARSE = [(96, 72)]   # this input size is (384 * 288)
_C.TRAIN.DATASET.OUT_RES_COARSE = [(120, 160)]   # this input size is (640 * 480)
# _C.TRAIN.DATASET.OUT_RES_COARSE = [(60, 80)]   # this input size is (640 * 480) # 在头部加一层下采样
_C.TRAIN.DATASET.OUT_RES_REFINE= [48, 96]
_C.TRAIN.DATASET.CUT_OUT_SIZE = 192
# _C.TRAIN.DATASET.CUT_OUT_SIZE = [(192, 192)] 
_C.TRAIN.DATASET.BASE_SIGMA= 2.0
_C.TRAIN.DATASET.BATCH_SIZE= 8
_C.TRAIN.DATASET.NUM_JOINTS= _C.NUM_JOINTS
_C.TRAIN.DATASET.NUM_WORKER= 4
_C.TRAIN.DATASET.augments = [1,2,4,5,6]
_C.TRAIN.DATASET.EPOCH_REMOVE_AUG = 140

# testing
_C.TEST = CN()
_C.TEST.DATASET = CN()
_C.TEST.DATASET.DATASET = 'syt_rag'
_C.TEST.DATASET.ROOT = '/home/rudolfmaxx/syt_kpt_python/kpt_data/test'
_C.TEST.DATASET.NAME = 'ori'
_C.TEST.DATASET.NUM_JOINTS = _C.NUM_JOINTS

_C.TEST.DATASET.OUT_RES_COARSE = _C.TRAIN.DATASET.OUT_RES_COARSE
_C.TEST.DATASET.OUT_RES_REFINE= _C.TRAIN.DATASET.OUT_RES_REFINE
_C.TEST.DATASET.CUT_OUT_SIZE= _C.TRAIN.DATASET.CUT_OUT_SIZE
_C.TEST.DATASET.BATCH_SIZE= 8
_C.TEST.DATASET.NUM_WORKER= 4
# size of images for each device
# _C.TEST.BATCH_SIZE = 32
_C.TEST.IMAGES_PER_GPU = 1
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.ADJUST = True
_C.TEST.REFINE = True
_C.TEST.SCALE_FACTOR = [1]
# group
_C.TEST.DETECTION_THRESHOLD = 0.2
_C.TEST.TAG_THRESHOLD = 1.
_C.TEST.USE_DETECTION_VAL = True
_C.TEST.IGNORE_TOO_MUCH = False
_C.TEST.MODEL_FILE = ''
_C.TEST.IGNORE_CENTER = True
_C.TEST.NMS_KERNEL = 3
_C.TEST.NMS_PADDING = 1
_C.TEST.PROJECT2IMAGE = False

_C.TEST.LOG_PROGRESS = False




# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = True
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = True
_C.DEBUG.SAVE_HEATMAPS_PRED = True
_C.DEBUG.SAVE_TAGMAPS_PRED = True
_C.DEBUG.SAVE_MASK_PRED = True


def _make_iterable(var):
    if not isinstance(var, (list, tuple)):
        var = (var)

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    _make_iterable(cfg.TRAIN.DATASET.OUT_RES_COARSE)    
    _make_iterable(cfg.TRAIN.DATASET.OUT_RES_REFINE)    
    cfg.freeze()



if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
