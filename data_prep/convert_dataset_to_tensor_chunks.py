import os
from pathlib import Path
import torchvision
import torch
from glob import glob
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description='Copy a file from one directory to another.')
parser.add_argument('--target_dir', default=r'./tensor_dataset', type=str, help='The path to the destination file')
args = parser.parse_args()

source_dir = Path('/gpfs/space/projects/rally2023/rally-estonia-cropped-antialias')
#source_dir = Path('./data_stuff/rally-estonia-cropped-antialias')
target_dir = Path(args.target_dir)
data_dirs = os.listdir(source_dir)
data_dirs = [Path(directory) for directory in data_dirs]
data_dirs = [source_dir / dir_name for dir_name in data_dirs]

os.makedirs(target_dir, exist_ok=True)

def process_directory(data_dir):
    frames_df_source = data_dir / 'nvidia_frames.csv'
    path_name = data_dir.stem
    print(f'Working on {path_name}')

    path_target_dir = target_dir / path_name
    os.makedirs(path_target_dir, exist_ok=True)
    frames_df_target = path_target_dir / 'nvidia_frames.csv'
    shutil.copy(frames_df_source, frames_df_target)

    image_dirs = glob(str(data_dir) + '/front_wide/*')
    dataset_tensor = torch.empty(len(image_dirs), 3, 68, 264)
    for i, img_dir in enumerate(image_dirs):
        # if i % 500 == 0:
        #     print(f'Progress {i*100/len(image_dirs):.2f}%')
        image = torchvision.io.read_image(img_dir)
        dataset_tensor[i] = image

    tensor_target = path_target_dir / 'full_img_tensor_TCHW.pth'
    torch.save(dataset_tensor, tensor_target)
    return path_name

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_directory, data_dir) for data_dir in data_dirs]
    for future in as_completed(futures):
        path_name = future.result()
        print(f'Completed {path_name}')