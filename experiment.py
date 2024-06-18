import argparse
import sys
import logging
import random
import os
import wandb

from pathlib import Path
from models.pilotnet import PilotNet
from models.perciever import Perceiver
from models.perciever_rnn import MLPPredictor, PerceiverRNN
from train.trainer import PerceiverTrainer, PilotNetTrainer
from torch.nn import MSELoss, L1Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data_prep.nvidia import NvidiaDataset, NvidiaDatasetRNN


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
        choices=['mse', 'mae'],
        default='mae',
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
        help="Maximum number of epochs to train."
    )
    
    argparser.add_argument(
        '--seq-length',
        type=int,
        default=8,
        help="Number of frames in one RNN sequence."
    )
    
    argparser.add_argument(
        '--stride',
        type=int,
        default=4,
        help="Stride between frame sequences."
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
        self.seq_length = args.seq_length
        self.stride = args.stride
        self.fps = 30


class TuneHyperparametersConfig:
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
        self.seq_length = args.seq_length
        self.stride = args.stride
        self.fps = 30

    def update(self, config):
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay


def train(train_config):

    if train_config.model_type == "pilotnet":
        model = PilotNet()
        trainer = PilotNetTrainer(train_config.model_name, wandb_project=train_config.wandb_project)
    elif train_config.model_type == "perceiver":
        pmodel = Perceiver(
            input_channels = 3,          # number of channels for each token of the input
            input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
            max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
            depth = 1,                   # depth of net. The shape of the final attention mechanism will be:
                                         #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 512,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 4,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            num_classes = 1,             # output number of classes
            attn_dropout = 0.,
            ff_dropout = 0.,
            weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = 2      # number of self attention blocks per cross attention
        )
        steering_classifier = MLPPredictor(512, 64)
        
        model = PerceiverRNN(pmodel, steering_classifier)
        trainer = PerceiverTrainer(train_config.model_name, wandb_project=train_config.wandb_project)
    else:
        logging.error("Unknown model type: %s", train_config.model_type)
        sys.exit()

    criterion = get_loss_function(train_config)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, betas=(0.9, 0.999),
                                  eps=1e-08, weight_decay=train_config.weight_decay, amsgrad=False)

    train_loader, valid_loader = load_data(train_config)

    trainer.train(model, train_config.model_type, train_loader, valid_loader, optimizer, criterion,
                  train_config.max_epochs, train_config.patience, train_config.learning_rate_patience, train_config.fps)


def get_loss_function(train_config):
    if train_config.loss == 'mse':
        return MSELoss()
    elif train_config.loss == 'mae':
        return L1Loss()
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
    elif train_config.model_type == "perceiver":
        train_dataset = NvidiaDatasetRNN(train_paths, train_config.seq_length, train_config.stride)
        valid_dataset = NvidiaDatasetRNN(valid_paths, train_config.seq_length, train_config.stride)
    else:
        logging.error("Unknown model type: %s", train_config.model_type)
        sys.exit()

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=False,
                              num_workers=train_config.num_workers, pin_memory=True,
                              persistent_workers=True, collate_fn=train_dataset.collate_fn)

    valid_loader = DataLoader(valid_dataset, batch_size=train_config.batch_size, shuffle=False,
                              num_workers=train_config.num_workers, pin_memory=True,
                              persistent_workers=False, collate_fn=train_dataset.collate_fn)

    return train_loader, valid_loader


def tune_hyperparameters(tune_hyperparameters_config):
    sweep_configuration = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'valid_loss'},
        'parameters': {
            'learning_rate': {'values': [1e-3, 1e-4, 1e-5]},
            'batch_size': {'values': [8, 16, 32, 64, 128, 256, 512]},
            'weight_decay': {'values': [1e-2, 1e-3, 1e-4]},
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=tune_hyperparameters_config.wandb_project)

    def sweep_train():
        with wandb.init(project=tune_hyperparameters_config.wandb_project):
            tune_hyperparameters_config.update(wandb.config)
            train_conf = TrainingConfig(tune_hyperparameters_config)
            train(train_conf)

            logging.info(f'Finishing wandb.')
            wandb.finish()

    wandb.agent(sweep_id, function=sweep_train)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    if args.mode == 'train':
        config = TrainingConfig(args)
        if config.wandb_project:
            wandb.init(project=config.wandb_project, config={
                "model_name": config.model_name,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                
            })
        train(config)
        if config.wandb_project:
            logging.info(f'Finishing wandb.')
            wandb.finish()
    elif args.mode == 'tune_hyperparameters':
        config = TuneHyperparametersConfig(args)
        tune_hyperparameters(config)

