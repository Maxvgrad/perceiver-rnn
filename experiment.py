import argparse
import sys
import logging
import random
import os
import wandb

from pathlib import Path
from models.pilotnet import PilotNet
from train.trainer import PilotNetTrainer
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data_prep.nvidia import NvidiaDataset


def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--model-name',
        required=False,
        default=None,
        help='Name of the model used for saving model and logging in W&B.'
    )

    argparser.add_argument(
        '--model-type',
        required=False,
        choices=['pilotnet'],
        default='pilotnet',
        help='Defines which model will be trained.'
    )

    argparser.add_argument(
        '--loss',
        required=False,
        choices=['mse'],
        default='mse',
        help='Loss function used for training.'
    )

    argparser.add_argument(
        '--dataset-folder',
        default="rally-estonia-cropped-antialias",
        help='Root path to the dataset.'
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
        help="Num of epochs after with learning rate will be retuced by factor of 0.1."
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
        help="Maximium number of epochs to train."
    )

    return argparser.parse_args()

class TrainingConfig:
    def __init__(self, args):
        self.model_type = args.model_type
        self.model_name = args.model_name
        self.loss = args.loss
        self.dataset_folder = args.dataset_folder
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.weight_decay = args.weight_decay
        self.learning_rate = args.learning_rate
        self.learning_rate_patience = args.learning_rate_patience
        self.wandb_project = args.wandb_project
        self.max_epochs = args.max_epochs
        self.patience = args.patience
        self.fps = 30


def main(train_config):
    if train_conf.wandb_project:
        wandb.init(project=train_conf.wandb_project)

    if train_config.model_type == "pilotnet":
        model = PilotNet()
        trainer = PilotNetTrainer(train_config.model_name, wandb_project=train_config.wandb_project)
    else:
        logging.error("Unknown model type: %s", train_config.model_type)
        sys.exit()

    criterion = get_loss_function(train_config)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, betas=(0.9, 0.999),
                                  eps=1e-08, weight_decay=train_config.weight_decay, amsgrad=False)

    train_loader, valid_loader = load_data(train_config)

    trainer.train(model, train_loader, valid_loader, optimizer, criterion,
                  train_config.max_epochs, train_config.patience, train_config.learning_rate_patience, train_config.fps)


def get_loss_function(train_config):
    if train_config.loss == 'mse':
        return MSELoss()
    else:
        logging.error("Unknown loss function type: %s", train_config.loss)
        sys.exit()


def load_data(train_config):
    logging.info("Reading from: %s", train_config.dataset_folder)

    dataset_path = Path(train_config.dataset_folder)
    random.seed(train_config.seed)
    data_dirs = os.listdir(dataset_path)
    random.shuffle(data_dirs)
    split_index = int(0.8 * len(data_dirs))
    train_paths = [dataset_path / dir_name for dir_name in data_dirs[:split_index]]
    valid_paths = [dataset_path / dir_name for dir_name in data_dirs[split_index:]]

    if train_config.model_type == "pilotnet":
        train_dataset = NvidiaDataset(train_paths)
        valid_dataset = NvidiaDataset(valid_paths)
    else:
        logging.error("Unknown model type: %s", train_config.model_type)
        sys.exit()

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True,
                              num_workers=train_config.num_workers, pin_memory=True,
                              persistent_workers=True, collate_fn=train_dataset.collate_fn)

    valid_loader = DataLoader(valid_dataset, batch_size=train_config.batch_size, shuffle=False,
                              num_workers=train_config.num_workers, pin_memory=True,
                              persistent_workers=False, collate_fn=train_dataset.collate_fn)

    return train_loader, valid_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    train_conf = TrainingConfig(args)
    main(train_conf)
