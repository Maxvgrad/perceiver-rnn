import argparse
import logging
import sys

from einops import rearrange
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy

import wandb
from datasets import build_dataset
from datasets.dataset_name import DatasetName
from models import build_model, build_criterion
from models.model_type import ModelType
from train.trainer import PerceiverTrainer, PilotNetTrainer


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
        choices=['mse', 'mae', 'ce'],
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


class TrainingConfig:
    def __init__(self, args):
        self.model_type = ModelType[args.model_type.upper()]
        self.model_name = args.model_name
        self.loss = args.loss
        self.dataset_folder = args.dataset_folder
        self.dataset_proportion = args.dataset_proportion
        if 'ucf11' in args.dataset_folder.lower():
            self.dataset_name = DatasetName.UCF_11
        else:
            self.dataset_name = DatasetName.RALLY_ESTONIA
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.weight_decay = args.weight_decay
        self.learning_rate = args.learning_rate
        self.learning_rate_patience = args.learning_rate_patience
        self.wandb_project = args.wandb_project
        self.max_epochs = args.max_epochs
        self.patience = args.patience
        self.augment = bool(args.augment)
        self.fps = 30
        self.clip_duration = args.clip_duration

        self.perceiver_seq_length = args.perceiver_seq_length
        self.perceiver_stride = args.perceiver_stride
        self.perceiver_img_pre_type = args.perceiver_img_pre_type
        self.perceiver_in_channels = args.perceiver_in_channels
        self.perceiver_latent_dim = args.perceiver_latent_dim
        self.perceiver_dropout = args.perceiver_dropout
        self.perceiver_depth = args.perceiver_depth
        self.perceiver_num_latents = args.perceiver_num_latents
        self.perceiver_cross_heads = args.perceiver_cross_heads
        self.perceiver_latent_heads = args.perceiver_latent_heads
        self.perceiver_cross_dim_head = args.perceiver_cross_dim_head
        self.perceiver_latent_dim_head = args.perceiver_latent_dim_head
        self.perceiver_self_per_cross_attn = args.perceiver_self_per_cross_attn

    def as_dict(self):
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'loss': self.loss,
            'dataset_folder': self.dataset_folder,
            'dataset_name': self.dataset_name,
            'dataset_proportion': self.dataset_proportion,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'weight_decay': self.weight_decay,
            'learning_rate': self.learning_rate,
            'learning_rate_patience': self.learning_rate_patience,
            'wandb_project': self.wandb_project,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'augment': self.augment,
            'fps': self.fps,
            'clip_duration': self.clip_duration,
            'perceiver_seq_length': self.perceiver_seq_length,
            'perceiver_stride': self.perceiver_stride,
            'perceiver_img_pre_type': self.perceiver_img_pre_type,
            'perceiver_in_channels': self.perceiver_in_channels,
            'perceiver_num_latents': self.perceiver_num_latents,
            'perceiver_latent_dim': self.perceiver_latent_dim,
            'perceiver_dropout': self.perceiver_dropout,
            'perceiver_depth': self.perceiver_depth,
            'perceiver_cross_heads': self.perceiver_cross_heads,
            'perceiver_latent_heads': self.perceiver_latent_heads,
            'perceiver_cross_dim_head': self.perceiver_cross_dim_head,
            'perceiver_latent_dim_head': self.perceiver_latent_dim_head,
            'perceiver_self_per_cross_attn': self.perceiver_self_per_cross_attn,
        }


class TuneHyperparametersConfig(TrainingConfig):
    def __init__(self, args):
        super().__init__(args)
        self.wandb_sweep_name = args.wandb_sweep_name

    def update(self, config):
        if hasattr(config, 'learning_rate'):
            self.learning_rate = config.learning_rate
        if hasattr(config, 'batch_size'):
            self.batch_size = config.batch_size
        if hasattr(config, 'weight_decay'):
            self.weight_decay = config.weight_decay
        if hasattr(config, 'augment'):
            self.augment = bool(config.augment)
        if hasattr(config, 'perceiver_seq_length'):
            self.perceiver_seq_length = config.perceiver_seq_length
        if hasattr(config, 'perceiver_stride'):
            self.perceiver_stride = config.perceiver_stride
        if hasattr(config, 'perceiver_in_channels'):
            self.perceiver_in_channels = config.perceiver_in_channels
        if hasattr(config, 'perceiver_latent_dim'):
            self.perceiver_latent_dim = config.perceiver_latent_dim
        if hasattr(config, 'perceiver_dropout'):
            self.perceiver_dropout = config.perceiver_dropout
        if hasattr(config, 'perceiver_depth'):
            self.perceiver_depth = config.perceiver_depth
        if hasattr(config, 'perceiver_num_latents'):
            self.perceiver_num_latents = config.perceiver_num_latents
        if hasattr(config, 'perceiver_cross_heads'):
            self.perceiver_cross_heads = config.perceiver_cross_heads
        if hasattr(config, 'perceiver_latent_heads'):
            self.perceiver_latent_heads = config.perceiver_latent_heads
        if hasattr(config, 'perceiver_cross_dim_head'):
            self.perceiver_cross_dim_head = config.perceiver_cross_dim_head
        if hasattr(config, 'perceiver_latent_dim_head'):
            self.perceiver_latent_dim_head = config.perceiver_latent_dim_head
        if hasattr(config, 'perceiver_self_per_cross_attn'):
            self.perceiver_self_per_cross_attn = config.perceiver_self_per_cross_attn


def train(args, train_config):
    train_loader, valid_loader = load_data(args)

    model = build_model(args)

    if train_config.model_type == ModelType.PILOTNET:
        trainer = PilotNetTrainer(train_config.model_name, wandb_project=train_config.wandb_project)
    elif train_config.model_type == ModelType.PERCEIVER:
        is_many_to_one = False
        save_model = True
        target_name = 'steering_angle'
        metric_multi_class_accuracy = None
        if train_config.dataset_name == DatasetName.UCF_11:
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

        elif train_config.dataset_name == DatasetName.RALLY_ESTONIA:

            def prepare_dataloader_data_fn_rally_estonia(loader_data):
                data, target_values, _ = loader_data
                inputs = rearrange(data['image'], 'b t c h w -> t b h w c')  # (T, B, H, W, C)
                target_values = rearrange(target_values, 'b t -> t b')  # (T, B)
                return inputs, target_values

            prepare_dataloader_data_fn = prepare_dataloader_data_fn_rally_estonia

        else:
            logging.error("Unsupported dataset.")
            sys.exit()

        trainer = PerceiverTrainer(
            prepare_dataloader_data_fn=prepare_dataloader_data_fn,
            is_many_to_one=is_many_to_one,
            model_name=train_config.model_name,
            wandb_project=train_config.wandb_project,
            target_name=target_name,
            save_model=save_model,
            metric_multi_class_accuracy=metric_multi_class_accuracy,
        )
    else:
        logging.error("Unknown model type: %s", train_config.model_type)
        sys.exit()

    criterion = build_criterion(args)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, betas=(0.9, 0.999),
                      eps=1e-08, weight_decay=train_config.weight_decay, amsgrad=False)

    trainer.train(model, train_config.model_type, train_loader, valid_loader, optimizer, criterion,
                  train_config.max_epochs, train_config.patience, train_config.learning_rate_patience, train_config.fps)


def load_data(args):
    logging.info("Dataset %s reading from: %s", args.dataset, args.dataset_folder)

    dataset_name = args.dataset.lower() if args.dataset is str else ''

    collate_fn = None
    if dataset_name == DatasetName.COCO_17.value:
        train_dataset = build_dataset(image_set='train', args=args)
        valid_dataset = build_dataset(image_set='val', args=args)
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


def tune_hyperparameters(tune_hyperparameters_config):
    sweep_configuration = {
        'method': 'random',
        'name': tune_hyperparameters_config.wandb_sweep_name,
        'metric': {'goal': 'minimize', 'name': 'valid_loss'},
        'parameters': {
            'perceiver_num_latents': {'values': [11, 16, 32, 256, 512]},
            'perceiver_latent_dim': {'values': [64, 128, 256, 512]},
            'perceiver_depth': {'values': [1, 2]},
            'perceiver_cross_heads': {'values': [1, 2, 4, 8]},
            'perceiver_latent_heads': {'values': [2, 4, 8]},
            'perceiver_cross_dim_head': {'values': [32, 64, 128]},
            'perceiver_latent_dim_head': {'values': [32, 64, 128]},
            'perceiver_self_per_cross_attn': {'values': [1, 2, 4]}
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 3,
            'eta': 3,
            'max_iter': tune_hyperparameters_config.max_epochs,
            'strict': True,
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=tune_hyperparameters_config.wandb_project)

    def sweep_train():
        with wandb.init(project=tune_hyperparameters_config.wandb_project):
            tune_hyperparameters_config.update(wandb.config)
            train_conf = TrainingConfig(tune_hyperparameters_config)
            train(wandb.config, train_conf)

            logging.info(f'Finishing wandb.')
            wandb.finish()

    wandb.agent(sweep_id, function=sweep_train)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    if args.mode == 'train':
        config = TrainingConfig(args)
        if config.wandb_project:
            wandb.init(project=config.wandb_project, config=config.as_dict())
        train(args, config)
        if config.wandb_project:
            logging.info(f'Finishing wandb.')
            wandb.finish()
    elif args.mode == 'tune_hyperparameters':
        config = TuneHyperparametersConfig(args)
        tune_hyperparameters(config)
