import logging
import sys

from einops import rearrange
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss
from torcheval.metrics import MulticlassAccuracy

from datasets.dataset_name import DatasetName
from .criterion import DetrLossCriterion
from .matcher import build_matcher, HungarianMatcher
from .model_type import ModelType
from .perciever import Perceiver
from .perciever_rnn import UcfClassPredictor, MLPPredictor, PerceiverRNN
from .pilotnet import PilotNet


def build_model(args):
    if args.model_type == ModelType.PILOTNET.value:
        return PilotNet()
    elif args.model_type == ModelType.PERCEIVER.value:
        is_many_to_one = False
        save_model = True
        target_name = 'steering_angle'
        metric_multi_class_accuracy = None
        if args.dataset == DatasetName.UCF_11.value:
            classifier_head = UcfClassPredictor(args.perceiver_latent_dim, num_classes=11)
            is_many_to_one = True
            save_model = False
            target_name = 'n/a'
            metric_multi_class_accuracy = MulticlassAccuracy()

            def prepare_dataloader_data_fn_ucf(loader_data):
                data = loader_data
                inputs = rearrange(data['video'], 'b c t h w -> t b h w c')
                target_values = data['label']
                return inputs, target_values

            prepare_dataloader_data_fn = prepare_dataloader_data_fn_ucf

        elif args.dataset == DatasetName.RALLY_ESTONIA.value:
            classifier_head = MLPPredictor(args.perceiver_latent_dim, 64)

            def prepare_dataloader_data_fn_rally_estonia(loader_data):
                data, target_values, _ = loader_data
                inputs = rearrange(data['image'], 'b t c h w -> t b h w c')  # (T, B, H, W, C)
                target_values = rearrange(target_values, 'b t -> t b')  # (T, B)
                return inputs, target_values

            prepare_dataloader_data_fn = prepare_dataloader_data_fn_rally_estonia

        else:
            logging.error("Unsupported dataset.")
            sys.exit()

        pmodel = Perceiver(
            input_channels=args.perceiver_in_channels,  # number of channels for each token of the input
            input_axis=2,  # number of axis for input data (2 for images, 3 for video)
            num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
            max_freq=10.,  # maximum frequency, hyperparameter depending on how fine the data is
            depth=args.perceiver_depth,  # depth of net. The shape of the final attention mechanism will be:
            #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=args.perceiver_num_latents,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=args.perceiver_latent_dim,  # latent dimension
            cross_heads=args.perceiver_cross_heads,  # number of heads for cross attention. paper said 1
            latent_heads=args.perceiver_latent_heads,  # number of heads for latent self attention, 8
            cross_dim_head=args.perceiver_cross_dim_head,  # number of dimensions per cross attention head
            latent_dim_head=args.perceiver_latent_dim_head,  # number of dimensions per latent self attention head
            num_classes=1,  # NOT USED. output number of classes.
            attn_dropout=args.perceiver_dropout,
            ff_dropout=args.perceiver_dropout,
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data=True,
            # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn=args.perceiver_self_per_cross_attn,  # number of self attention blocks per cross attention
            final_classifier_head=False  # mean pool and project embeddings to number of classes (num_classes) at the end
        )

        return PerceiverRNN(pmodel, classifier_head, preprocess=args.perceiver_img_pre_type)

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

        matcher = HungarianMatcher()

        losses = ['labels', 'boxes', 'cardinality']

        weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
        weight_dict['loss_giou'] = 2

        return DetrLossCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weights=weight_dict,
            eos_coef=0.1,
            losses=losses
        )
    else:
        logging.error("Unknown loss function type: %s", args.loss)
        sys.exit()