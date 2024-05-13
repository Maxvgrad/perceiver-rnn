from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchvision
import sys
from torch.utils.data._utils.collate import default_collate


class Normalize(object):
    def __call__(self, data, transform=None):
        image = data["image"]
        image = image / 255
        data["image"] = image
        return data
    
class NvidiaDataset(Dataset):

    def __init__(self, dataset_paths, transform=None, camera="front_wide", name="Nvidia dataset",
                 filter_turns=False, metadata_file="nvidia_frames.csv", color_space="rgb", dataset_proportion=1.0):
        self.name = name
        self.metadata_file = metadata_file
        self.color_space = color_space
        self.dataset_paths = dataset_paths
        if transform:
            self.transform = transform
            print(f'[NvidiaDataset] Using transform from argument:', self.transform)
        else:
            self.transform = transforms.Compose([Normalize()])
            print(f'[NvidiaDataset] Using default transform:', self.transform)

        self.camera_name = camera
        self.target_size = 1

        datasets = [self.read_dataset(dataset_path, camera) for dataset_path in self.dataset_paths]
        self.frames = pd.concat(datasets)
        keep_n_frames = np.ceil(len(self.frames) * dataset_proportion)
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
            'steering_angle': np.array(frame["steering_angle"]),
            'vehicle_speed': np.array(frame["vehicle_speed"]),
            'autonomous': np.array(frame["autonomous"]),
            'turn_signal': np.array(frame["turn_signal"]),
            'row_id': np.array(frame["row_id"]),
            'timestamp': np.array(frame["index"]).astype('datetime64[ns]').astype(np.float64),
        }

        target_values = frame["steering_angle"]

        if self.transform:
            data = self.transform(data)

        target = np.zeros((1, self.target_size))
        target[0, :] = target_values
        conditional_mask = np.ones((1, self.target_size))

        return data, target.reshape(-1), conditional_mask.reshape(-1)

    def __len__(self):
        return len(self.frames.index)

    def read_dataset(self, dataset_path, camera):
        frames_df = pd.read_csv(dataset_path / self.metadata_file)
        len_before_filtering = len(frames_df)
            
        frames_df["row_id"] = frames_df.index

        frames_df = frames_df[frames_df['steering_angle'].notna()]  # TODO: one steering angle is NaN, why?
        frames_df = frames_df[frames_df['vehicle_speed'].notna()]
        frames_df = frames_df[frames_df[f'{camera}_filename'].notna()]

        frames_df["turn_signal"].fillna(1, inplace=True)
        frames_df["turn_signal"] = frames_df["turn_signal"].astype(int)

        # Removed frames marked as skipped
        frames_df = frames_df[frames_df["turn_signal"] != -1]  # TODO: remove magic values.

        len_after_filtering = len(frames_df)

        camera_images = frames_df[f"{camera}_filename"].to_numpy()
        frames_df["image_path"] = [str(dataset_path / image_path) for image_path in camera_images]

        print(f"{dataset_path}: length={len(frames_df)}, filtered={len_before_filtering-len_after_filtering}")
        frames_df.reset_index(inplace=True)
        return frames_df
