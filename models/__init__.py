import logging
import sys

from torch.nn import MSELoss, L1Loss, CrossEntropyLoss

from datasets.dataset_name import DatasetName
from .criterion import DetrLossCriterion
from .matcher import build_matcher, HungarianMatcher
from .model_type import ModelType
from .perciever import Perceiver
from .perciever_rnn import UcfClassPredictor, MLPPredictor, PerceiverRnn, build_perceiver_rnn
from .pilotnet import PilotNet
from .postprocessor import PostProcess


def build_model(args):
    if args.model_type == ModelType.PILOTNET.value:
        return PilotNet()
    elif args.model_type == ModelType.PERCEIVER.value:
        return build_perceiver_rnn(args)
    else:
        logging.error("Unknown model type: %s", args.model_type)
        sys.exit()


def build_criterion(args):
    if args.loss == 'mse':
        return MSELoss()
    elif args.loss == 'mae':
        return L1Loss()
    elif args.loss == 'ce':
        return CrossEntropyLoss()
    elif args.loss == 'detr':
        # the `num_classes` naming here is somewhat misleading.
        # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
        # is the maximum id for a class in your dataset. For example,
        # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
        # As another example, for a dataset that has a single class with id 1,
        # you should pass `num_classes` to be 2 (max_obj_id + 1).
        # For more details on this, check the following discussion
        # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
        num_classes = 91 #TODO: based on dataset

        hungarian_matcher = HungarianMatcher()
        losses = ['labels']#, 'boxes', 'cardinality']
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        return DetrLossCriterion(
            num_classes=num_classes,
            matcher=hungarian_matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=losses
        )
    else:
        logging.error("Unknown loss function type: %s", args.loss)
        sys.exit()


def build_postprocessors(args):
    postprocessors = {}
    if args.dataset == DatasetName.COCO_17.value:
        postprocessors['bbox'] = PostProcess()
    return postprocessors