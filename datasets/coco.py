from pathlib import Path

import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms.v2 as T
from torchvision.datasets import wrap_dataset_for_transforms_v2


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file, transforms)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        return img, target


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.ToImage(),
            T.CenterCrop(224),
            T.Resize(224),
            T.SanitizeBoundingBoxes(),
            T.ToDtype(torch.float32, scale=True),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.ToImage(),
            T.CenterCrop(224),
            T.Resize(224),
            T.SanitizeBoundingBoxes(),
            T.ToDtype(torch.float32, scale=True),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.dataset_folder)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set))
    dataset = wrap_dataset_for_transforms_v2(dataset)
    return dataset
