import argparse
import copy
import logging
import sys
from collections import namedtuple

import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import build_dataset
from datasets.dataset_name import DatasetName
from metrics import get_build_evaluators_fn
from models import build_model, build_criterion, build_postprocessors
from train import build_trainer


def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--model-name',
        required=False,
        help='Name of the model used for saving model and logging in W&B.'
    )

    argparser.add_argument(
        '--mode',
        required=True,
        choices=['train', 'tune_hyperparameters'],
        help="Mode to run in."
    )

    argparser.add_argument(
        '--model-type',
        required=False,
        choices=['pilotnet', 'perceiver'],
        default='pilotnet',
        help='Defines which model will be trained.'
    )

    argparser.add_argument(
        '--loss',
        required=False,
        choices=['detr', 'mse', 'mae', 'ce'],
        default='mse',
        help='Loss function used for training.'
    )

    argparser.add_argument(
        '--dataset-folder',
        help='Root path to the dataset.'
    )

    argparser.add_argument(
        '--dataset',
        required=False,
        choices=['coco17', 'ucf11', 'rally-estonia'],
        default='coco17',
        help='Dataset name.'
    )

    argparser.add_argument(
        '--dataset-proportion',
        type=float,
        default=1.0,
        help="Dataset proportion taken."
    )

    argparser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed."
    )

    argparser.add_argument(
        '--wandb-project',
        required=False,
        help='W&B project name to use for metrics. Wandb logging is disabled when no project name is provided.'
    )

    argparser.add_argument(
        '--wandb-sweep-name',
        required=False,
        default='sweep-demo',
        help='W&B sweep name.'
    )

    argparser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Weight decay used in training.'
    )

    argparser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='Number of workers used for data loading.'
    )

    argparser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help="Learning rate used in training."
    )

    argparser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-02,
        help='Weight decay used in training.'
    )

    argparser.add_argument(
        '--learning-rate-patience',
        type=int,
        default=10,
        help="Num of epochs after with learning rate will be reduced by factor of 0.1."
    )

    argparser.add_argument(
        '--patience',
        type=int,
        default=10,
        help="Number of epochs to train without improvement in validation loss. Used for early stopping."
    )

    argparser.add_argument(
        '--max-epochs',
        type=int,
        default=100,
        help="Maximum number of epochs to train."
    )

    argparser.add_argument(
        '--augment',
        type=bool,
        default=False,
        help="Augment."
    )

    argparser.add_argument(
        '--clip-duration',
        type=float,
        default=2.0,
        help="Duration of sampled clip for each video."
    )

    argparser.add_argument(
        '--perceiver-seq-length',
        type=int,
        default=8,
        help="Number of frames in one RNN sequence."
    )

    argparser.add_argument(
        '--perceiver-stride',
        type=int,
        default=4,
        help="Stride between frame sequences."
    )

    argparser.add_argument(
        '--perceiver-img-pre-type',
        type=str,
        choices=['cnn', 'resnet18', 'dino'],
        default='cnn',
        help="Perceiver image preprocess model"
    )

    argparser.add_argument(
        '--perceiver-in-channels',
        type=int,
        default=3,
        help="Perceiver input channels."
    )
    argparser.add_argument(
        '--perceiver-latent-dim',
        type=int,
        default=512,
    )

    argparser.add_argument(
        '--perceiver-num-latents',
        type=int,
        default=256,
        help="Number of latents, or induced set points, or centroids. "
             "Different papers giving it different names."
    )

    argparser.add_argument(
        '--perceiver-dropout',
        type=float,
        default=0,
    )

    argparser.add_argument(
        '--perceiver-depth',
        type=int,
        default=1,
        help="Depth of net."
    )

    argparser.add_argument(
        '--perceiver-cross-heads',
        type=int,
        default=1,
        help="Number of heads for cross attention."
    )

    argparser.add_argument(
        '--perceiver-latent-heads',
        type=int,
        default=8,
        help="Number of heads for latent self attention."
    )

    argparser.add_argument(
        '--perceiver-cross-dim-head',
        type=int,
        default=64,
        help="Number of dimensions per cross attention head."
    )

    argparser.add_argument(
        '--perceiver-latent-dim-head',
        type=int,
        default=64,
        help="Number of dimensions per latent self attention head."
    )

    argparser.add_argument(
        '--perceiver-self-per-cross-attn',
        type=int,
        default=2,
        help="Number of self attention blocks per cross attention."
    )

    return argparser.parse_args()


def train(args):
    train_loader, valid_loader = load_data(args)
    model = build_model(args)
    trainer = build_trainer(args)
    postprocessors = build_postprocessors(args)
    build_evaluators_fn = get_build_evaluators_fn(args, valid_loader.dataset)
    criterion = build_criterion(args)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                      eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    trainer.train(model, args.model_type, train_loader, valid_loader, optimizer, criterion, postprocessors,
                  build_evaluators_fn, args.max_epochs, args.patience, args.learning_rate_patience)


def load_data(args):
    logging.info("Dataset %s reading from: %s", args.dataset, args.dataset_folder)

    dataset_name = args.dataset.lower() if args.dataset is not None else ''

    collate_fn = None
    if dataset_name == DatasetName.COCO_17.value:
        train_dataset = build_dataset(image_set='train', args=args)
        valid_dataset = build_dataset(image_set='val', args=args)

        def collate_fn(batch):
            import torch
            images = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            images = torch.stack(images, dim=0)
            targets = targets
            return images, targets

        collate_fn = collate_fn
    elif dataset_name == DatasetName.UCF_11.value:
        train_dataset, valid_dataset = build_dataset(image_set=None, args=args)
    elif dataset_name == DatasetName.RALLY_ESTONIA.value:
        train_dataset, valid_dataset = build_dataset(image_set=None, args=args)
        collate_fn = train_dataset.collate_fn
    else:
        logging.error("Unknown dataset name: %s", args.dataset_name)
        sys.exit()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=True, collate_fn=collate_fn)

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=False, collate_fn=collate_fn)

    return train_loader, valid_loader


def tune_hyperparameters(args):
    sweep_configuration = {
        'method': 'random',
        'name': args.wandb_sweep_name,
        'metric': {'goal': 'minimize', 'name': 'valid_loss'},
        'parameters': {
            'perceiver_num_latents': {'values': [64, 128]},
            'perceiver_latent_dim': {'values': [32, 64, 128]},
            'perceiver_depth': {'values': [1, 2]},
            'perceiver_cross_heads': {'values': [1, 2, 4, 8]},
            'perceiver_latent_heads': {'values': [1, 2, 4, 8]},
            'perceiver_self_per_cross_attn': {'values': [1, 2, 4]}
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 3,
            'eta': 3,
            'max_iter': 5,
            'strict': True,
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.wandb_project)

    def sweep_train():
        with wandb.init(project=args.wandb_project):

            wandb_config_copy = copy.deepcopy(wandb.config)

            wandb_config_copy['perceiver_cross_dim_head'] = args.perceiver_in_channels // wandb_config_copy['perceiver_cross_heads']
            wandb_config_copy['perceiver_latent_dim_head'] = wandb_config_copy['perceiver_latent_dim'] // wandb_config_copy['perceiver_latent_heads']

            new_args = merge_args_with_wandb_config(args, wandb_config_copy)

            train(new_args)
            logging.info(f'Finishing wandb.')
            wandb.finish()

    wandb.agent(sweep_id, function=sweep_train)


def merge_args_with_wandb_config(args, wandb_config):
    args_dict = vars(args)  # Convert Namespace to a dictionary
    for key, value in wandb_config.items():
        if key in args_dict:
            args_dict[key] = value
    return namedtuple('Args', args_dict.keys())(*args_dict.values())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    if args.mode == 'train':
        if args.wandb_project:
            wandb.init(project=args.wandb_project, config=args)
        train(args)
        if args.wandb_project:
            logging.info(f'Finishing wandb.')
            wandb.finish()
    elif args.mode == 'tune_hyperparameters':
        tune_hyperparameters(args)
