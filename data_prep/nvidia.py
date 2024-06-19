from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchvision
import torch
import sys
from torch.utils.data._utils.collate import default_collate
from .custom_transforms import ImageTransform, Normalize

class NvidiaDataset(Dataset):

    def __init__(self, dataset_paths, transform=False, camera="front_wide", name="Nvidia dataset",
                 filter_turns=False, metadata_file="nvidia_frames.csv", color_space="rgb", dataset_proportion=1.0):
        self.name = name
        self.metadata_file = metadata_file
        self.color_space = color_space
        self.dataset_paths = dataset_paths
        if not transform:
            self.transform = transforms.Compose([Normalize()])
            print(f'[NvidiaDataset] Using transform from argument:', self.transform)
        else:
            self.transform = transforms.Compose([ImageTransform()])
                    
            print(f'[NvidiaDataset] Using default transform:', self.transform)

        self.camera_name = camera
        self.target_size = 1

        datasets = [self.read_dataset(dataset_path, camera) for dataset_path in self.dataset_paths]
        self.datasets = datasets

        for i, dataset in enumerate(datasets):
            dataset['path_id'] = i
        
        self.frames = pd.concat(datasets).reset_index(drop=True)
        keep_n_frames = np.ceil(len(self.frames) * dataset_proportion).astype(int)
        self.frames = self.frames.head(keep_n_frames)

        if filter_turns:
            print("Filtering turns with blinker signal")
            self.frames = self.frames[self.frames.turn_signal == 1]
            
    def collate_fn(self, batch):
        data, targets, _ = default_collate(batch)
        
        return data, targets, _

    def __getitem__(self, idx):
        frame = self.frames.iloc[idx]
        if self.color_space == "rgb":
            image = torchvision.io.read_image(frame["image_path"])
        else:
            print(f"Unknown color space: ", self.color_space)
            sys.exit()

        data = {
            'image': image,
            'path_id': np.array(frame['path_id']),
            'steering_angle': np.array(frame["steering_angle"]),
            'vehicle_speed': np.array(frame["vehicle_speed"]),
            'autonomous': np.array(frame["autonomous"]),
            'turn_signal': np.array(frame["turn_signal"]),
            'row_id': np.array(frame["row_id"]),
            'timestamp': np.array(frame["index"]).astype('datetime64[ns]').astype(np.float64),
        }

        target_values = np.array(frame["steering_angle"], dtype=np.float32)

        if self.transform:
            data = self.transform(data)

        conditional_mask = np.ones((1, self.target_size))

        return data, target_values.reshape(-1), conditional_mask.reshape(-1)

    def __len__(self):
        return len(self.frames.index)

    def read_dataset(self, dataset_path, camera):
        frames_df = pd.read_csv(dataset_path / self.metadata_file)
        len_before_filtering = len(frames_df)
            
        frames_df["row_id"] = frames_df.index

        frames_df = frames_df[frames_df['steering_angle'].notna()]  # TODO: one steering angle is NaN, why?
        frames_df = frames_df[frames_df['vehicle_speed'].notna()]
        frames_df = frames_df[frames_df[f'{camera}_filename'].notna()]

        frames_df.fillna({'turn_signal': 1}, inplace=True)
        frames_df["turn_signal"] = frames_df["turn_signal"].astype(int)

        # Removed frames marked as skipped
        frames_df = frames_df[frames_df["turn_signal"] != -1]  # TODO: remove magic values.

        len_after_filtering = len(frames_df)

        camera_images = frames_df[f"{camera}_filename"].to_numpy()
        frames_df["image_path"] = [str(dataset_path / image_path) for image_path in camera_images]

        print(f"{dataset_path}: length={len(frames_df)}, filtered={len_before_filtering-len_after_filtering}")
        frames_df.reset_index(drop=True, inplace=True)
        return frames_df


class NvidiaDatasetRNN(NvidiaDataset): 
    def __init__(self, dataset_paths, seq_length, seq_stride, 
                 transform=None, camera="front_wide", 
                 name="Nvidia dataset", filter_turns=False, 
                 metadata_file="nvidia_frames.csv", 
                 color_space="rgb", dataset_proportion=1):
        "shape B, T, C, H, W"
        super().__init__(dataset_paths, transform, camera, name,
                          filter_turns, metadata_file, color_space, dataset_proportion)
        self.sequence_ids = self.create_sequence_indices(seq_length, seq_stride)
        
        self.target_size = seq_length

    def collate_fn(self, batch):
        data, targets, _ = default_collate(batch)
        
        return data, targets, _
    
    def create_sequence_indices(self, sequence_length, stride):
        # generates IDs for frame DF for each possible sequence
        sequence_ids = []
        for path_id in self.frames.path_id.unique():
            path_df = self.frames[self.frames.path_id == path_id]
            num_sequences = (len(path_df) - sequence_length) // stride + 1
            for i in range(num_sequences): 
                start = i * stride
                sequence_ids.append(np.array(path_df.index)[start:start+sequence_length])
        return sequence_ids
    

    def __getitem__(self, idx):
        sequence_ids = self.sequence_ids[idx]
        sequence_df = self.frames.loc[sequence_ids]
        sequence_images = []
        for image_path in sequence_df.image_path:
            if self.color_space == "rgb":
                image = torchvision.io.read_image(image_path)
                sequence_images.append(image)
            else:
                print(f"Unknown color space: ", self.color_space)
                sys.exit()
        sequence_images = torch.stack(sequence_images)
        
        sequence_data = {
            'image': sequence_images,
            'path_id': np.array(sequence_df['path_id']),
            'steering_angle': np.array(sequence_df["steering_angle"]),
            'vehicle_speed': np.array(sequence_df["vehicle_speed"]),
            'autonomous': np.array(sequence_df["autonomous"]),
            'turn_signal': np.array(sequence_df["turn_signal"]),
            'row_id': np.array(sequence_df["row_id"]),
            'timestamp': np.array(sequence_df["index"]).astype('datetime64[ns]').astype(np.float64),
        }

        target_values = np.array(sequence_df["steering_angle"])

        if self.transform:
            sequence_data = self.transform(sequence_data)

        target = np.zeros((1, self.target_size))
        target[0, :] = target_values
        conditional_mask = np.ones((1, self.target_size))
        
        return sequence_data, target.reshape(-1), conditional_mask.reshape(-1)

    def __len__(self):
        return len(self.sequence_ids)


