from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchvision
import torch
import sys
from torch.utils.data._utils.collate import default_collate
from .custom_transforms import ImageTransform, Normalize


class NvidiaDatasetOptim(Dataset):

    def __init__(self, dataset_paths, transform=False, name="Nvidia dataset optimized",
                 filter_turns=False, metadata_file="nvidia_frames.csv",
                 full_tensor_filename='full_img_tensor_TCHW.pth', 
                 dataset_proportion=1.0):
        self.name = name
        self.metadata_file = metadata_file
        self.dataset_paths = dataset_paths
        self.full_tensor_filename = full_tensor_filename

        if not transform:
            self.transform = transforms.Compose([Normalize()])
            print(f'[NvidiaDataset] Using transform from argument:', self.transform)
        else:
            self.transform = transforms.Compose([ImageTransform()])
            print(f'[NvidiaDataset] Using default transform:', self.transform)

        self.target_size = 1

        self.datasets = [self.read_dataset(dataset_path) for dataset_path in self.dataset_paths]

        for i, dataset in enumerate(self.datasets):
            dataset['path_id'] = i
        
        self.frames = pd.concat(self.datasets).reset_index(drop=True)
        keep_n_frames = np.ceil(len(self.frames) * dataset_proportion).astype(int)
        self.frames = self.frames.head(keep_n_frames)

        if filter_turns:
            print("Filtering turns with blinker signal")
            self.frames = self.frames[self.frames.turn_signal == 1]
        
        self.full_tensor_cache = {'path': None,
                                   'full_img_tensor': None}

    def read_dataset(self, dataset_path):
        frames_df = pd.read_csv(dataset_path / self.metadata_file)
        len_before_filtering = len(frames_df)
            
        frames_df["image_index"] = frames_df.index
        
        frames_df = frames_df[frames_df['steering_angle'].notna()]  # TODO: one steering angle is NaN, why?
        frames_df = frames_df[frames_df['vehicle_speed'].notna()]

        frames_df.fillna({'turn_signal': 1}, inplace=True)
        frames_df["turn_signal"] = frames_df["turn_signal"].astype(int)

        # Removed frames marked as skipped
        frames_df = frames_df[frames_df["turn_signal"] != -1]  # TODO: remove magic values.

        len_after_filtering = len(frames_df)

        frames_df["full_tensor_path"] = [str(dataset_path / 'full_path_tensor_TCHW.pth') for _ in range(len(frames_df))]

        print(f"{dataset_path}: length={len(frames_df)}, filtered={len_before_filtering-len_after_filtering}")
        frames_df.reset_index(drop=True, inplace=True)
        return frames_df

    def collate_fn(self, batch):
        data, targets, _ = default_collate(batch)
        
        return data, targets, _

    def __getitem__(self, idx):
        frame = self.frames.iloc[idx]
        full_tensor_path = frame.full_tensor_path
        image_index = frame.image_index
        if full_tensor_path == self.full_tensor_cache['path']:
            full_img_tensor = self.full_tensor_cache['full_img_tensor']
        else: 
            full_img_tensor = torch.load(full_tensor_path)
            self.full_tensor_cache['path'] = full_tensor_path
            self.full_tensor_cache['full_img_tensor'] = full_img_tensor
            
        image = full_img_tensor[image_index]
        data = {
            'image': image,
            'path_id': np.array(frame['path_id']),
            'steering_angle': np.array(frame["steering_angle"]),
            'vehicle_speed': np.array(frame["vehicle_speed"]),
            'autonomous': np.array(frame["autonomous"]),
            'turn_signal': np.array(frame["turn_signal"]),
            'image_index': np.array(frame["image_index"]),
            'timestamp': np.array(frame["index"]).astype('datetime64[ns]').astype(np.float64),
        }

        target_values = np.array(frame["steering_angle"], dtype=np.float32)

        if self.transform:
            data = self.transform(data)

        conditional_mask = np.ones((1, self.target_size))

        return data, target_values.reshape(-1), conditional_mask.reshape(-1)

    def __len__(self):
        return len(self.frames.index)


class NvidiaDatasetRNNOptim(NvidiaDatasetOptim): 
    def __init__(self, dataset_paths, seq_length, seq_stride, 
                 transform=None, **kwargs):
        "shape B, T, C, H, W"
        super().__init__(dataset_paths, transform=transform, **kwargs)
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
        sequence_img_ids = np.array(sequence_df.image_index)
        sequence_images = []
        full_tensor_path = sequence_df.full_tensor_path.iloc[0]

        assert all(sequence_df.full_tensor_path == full_tensor_path)

        if full_tensor_path == self.full_tensor_cache['path']:
            full_img_tensor = self.full_tensor_cache['full_img_tensor']
        else: 
            full_img_tensor = torch.load(full_tensor_path)
            self.full_tensor_cache['path'] = full_tensor_path
            self.full_tensor_cache['full_img_tensor'] = full_img_tensor
        
        print(full_img_tensor)
        sequence_images = full_img_tensor[sequence_img_ids]

        sequence_data = {
            'image': sequence_images,
            'path_id': np.array(sequence_df['path_id']),
            'steering_angle': np.array(sequence_df["steering_angle"]),
            'vehicle_speed': np.array(sequence_df["vehicle_speed"]),
            'autonomous': np.array(sequence_df["autonomous"]),
            'turn_signal': np.array(sequence_df["turn_signal"]),
            'image_index': np.array(sequence_df["image_index"]),
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

