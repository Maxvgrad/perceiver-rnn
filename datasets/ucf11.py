import logging
import sys
from pathlib import Path
from random import shuffle
from typing import Type, Optional, Callable, Dict, Any

import torch
from pytorchvideo.data import LabeledVideoDataset, ClipSampler
from torch.utils.data import RandomSampler, Sampler


def Ucf11(
        clip_sampler: ClipSampler,
        video_sampler: Type[Sampler] = RandomSampler,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        data_path: str = './',
        video_path_prefix: str = "",
        decode_audio: bool = False,
        decoder: str = "pyav",
) -> tuple[LabeledVideoDataset, LabeledVideoDataset]:
    """
    A helper function to create ``LabeledVideoDataset`` object for the Ucf101 dataset.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video.

    """

    def make_ucf11_dataset(data_path):
        data_path = Path(data_path)
        if not data_path.exists():
            logging.error("Download ucf11 dataset is not supported.")
            sys.exit()
            # download_and_unzip(_ucf11_url, root, False)

        # Collect all class names, scene folders, and label2id mapping
        classes = sorted(x.name for x in data_path.glob("*") if x.is_dir())
        label2id = {}
        scene_folders = []
        for class_id, class_name in enumerate(classes):
            label2id[class_name] = class_id
            class_folder = data_path / class_name
            scene_folders.extend(list(filter(Path.is_dir, class_folder.glob('v_*'))))

        shuffle(scene_folders)

        num_train_scenes = int(0.8 * len(scene_folders))
        train_paths, val_paths = [], []
        for i, scene in enumerate(scene_folders):
            class_id = label2id[scene.parent.name]
            labeled_paths = [(video, {'label': class_id}) for video in scene.glob('*.mpg')]
            if i < num_train_scenes:
                train_paths.extend(labeled_paths)
            else:
                val_paths.extend(labeled_paths)
        return train_paths, val_paths, label2id, classes

    train_paths, val_paths, label2id, classes = make_ucf11_dataset(data_path)

    train_dataset = LabeledVideoDataset(
        train_paths,
        clip_sampler=clip_sampler,
        video_sampler=video_sampler,
        decode_audio=decode_audio,
        transform=transform,
        decoder=decoder,
    )

    val_dataset = LabeledVideoDataset(
        val_paths,
        clip_sampler=clip_sampler,
        video_sampler=video_sampler,
        decode_audio=decode_audio,
        transform=transform,
        decoder=decoder,
    )

    return train_dataset, val_dataset



