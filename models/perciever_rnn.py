import logging
import sys

import torch.nn.functional as F
from einops.layers.torch import Reduce
from torch import nn, Tensor

from datasets.dataset_name import DatasetName
from models import Perceiver
from .backbone import build_image_preprocess_backbone


class MLPPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.linear_stack = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.linear_stack(x)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ObjectDetectionHead(nn.Module):
    def __init__(self, num_classes, num_latents, latent_dim):
        """ Initializes the model.
        Parameters:
            num_classes: number of object classes
            num_latents: number of object queries, ie detection slot. This is the maximal number of objects
                         model can detect in a single image. For COCO, we recommend 100 queries.
            latent_dim: dimension of the latent object query.
        """
        super().__init__()
        self.num_queries = num_latents
        self.class_embed = nn.Linear(latent_dim, num_classes + 1)
        self.bbox_embed = MLP(latent_dim, latent_dim, 4, 3)

    def forward(self, hs: Tensor):
        """Forward pass of the ObjectDetectionHead.
            Parameters:
                - hs: Tensor
                    Hidden states from the model, of shape [batch_size x num_queries x latent_dim].

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
        """
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out


class UcfClassPredictor(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.linear_stack = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.linear_stack(x)


class PerceiverRnn(nn.Module):
    def __init__(self, image_preprocess, perceiver, classifier_head):
        super().__init__()
        self.perceiver = perceiver
        self.image_preprocess = image_preprocess
        self.classifier_head = classifier_head

    def forward(self, batch, latents=None):
        # batch (B, H, W, C)
        # latents (B, num_latents, latent_dim)
        preprocessed = self.image_preprocess(batch)
        latents = self.perceiver.forward(preprocessed, latents=latents, return_embeddings=True)
        predictions = self.classifier_head.forward(latents)
        return predictions, latents


def build_perceiver_rnn(args):
    image_preprocess_backbone = build_image_preprocess_backbone(args)

    if args.dataset == DatasetName.COCO_17.value:
        classifier_head = ObjectDetectionHead(
            num_classes=91,
            num_latents=args.perceiver_num_latents,
            latent_dim=args.perceiver_latent_dim
        )
    elif args.dataset == DatasetName.UCF_11.value:
        classifier_head = UcfClassPredictor(args.perceiver_latent_dim, num_classes=11)
    elif args.dataset == DatasetName.RALLY_ESTONIA.value:
        classifier_head = MLPPredictor(args.perceiver_latent_dim, 64)
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

    return PerceiverRnn(
        image_preprocess=image_preprocess_backbone,
        perceiver=pmodel,
        classifier_head=classifier_head
    )

