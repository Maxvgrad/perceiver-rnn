import logging
import os
import random
import sys
from pathlib import Path

import torchvision
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.transforms import (ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample, UniformCropVideo,
                                     Normalize)
from torch.utils.data import SequentialSampler
from torch.utils.data import Subset

from .coco import build as build_coco
from .nvidia import NvidiaDataset, NvidiaDatasetRNN
from .ucf11 import Ucf11


def build_dataset(args, image_set):
    if args.dataset == 'coco17':
        return _dataset_proportion(args=args, dataset=build_coco(args, image_set))
    elif args.dataset == 'ucf11':
        return Ucf11(
            clip_sampler=make_clip_sampler('random', args.clip_duration),
            video_sampler=SequentialSampler,
            data_path=args.dataset_folder,
            transform=torchvision.transforms.Compose(
                [
                    ApplyTransformToKey(
                        key="video",
                        transform=torchvision.transforms.Compose([
                            ShortSideScale(size=224),
                            UniformTemporalSubsample(num_samples=int(60 * args.clip_duration)),
                            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])),
                    UniformCropVideo(size=224),
                ]
            ),
            dataset_proportion=args.dataset_proportion
        )
    elif args.dataset == 'rally-estonia':
        dataset_path = Path(args.dataset_folder)
        random.seed(args.seed)
        data_dirs = os.listdir(dataset_path)
        random.shuffle(data_dirs)
        split_index = int(0.8 * len(data_dirs))
        train_paths = [dataset_path / dir_name for dir_name in data_dirs[:split_index]]
        valid_paths = [dataset_path / dir_name for dir_name in data_dirs[split_index:]]

        if args.model_type == "pilotnet":
            train_dataset = NvidiaDataset(train_paths, dataset_proportion=args.dataset_proportion)
            valid_dataset = NvidiaDataset(valid_paths, dataset_proportion=args.dataset_proportion)
        elif args.model_type == "perceiver":
            train_dataset = NvidiaDatasetRNN(
                train_paths, args.perceiver_seq_length, args.perceiver_stride,
                dataset_proportion=args.dataset_proportion)
            valid_dataset = NvidiaDatasetRNN(
                valid_paths, args.perceiver_seq_length, args.perceiver_stride,
                dataset_proportion=args.dataset_proportion)
        else:
            logging.error("Unknown model type: %s", args.model_type)
            sys.exit()

        return train_dataset, valid_dataset
    raise ValueError(f'dataset {args.dataset_file} not supported')


def _dataset_proportion(args, dataset):
    if args.dataset_proportion < 1.0:
        dataset_size = len(dataset)
        subset_size = int(args.dataset_proportion * dataset_size)
        indices = list(range(dataset_size))
        random.seed(args.seed)
        random.shuffle(indices)
        subset_indices = indices[:subset_size]
        dataset = Subset(dataset, subset_indices)

    return dataset


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco