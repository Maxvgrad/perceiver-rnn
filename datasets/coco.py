from pathlib import Path

import torch
import torch.utils.data
import torch.utils.data
import torchvision.datasets
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.transforms import v2

from datasets.transforms import NormalizeBoxes


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file, transforms=transforms)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)

        if len(target) == 0:
            # COCO dataset contains images with no annotations
            # If so then provide empty target
            target = [
                {
                    'segmentation': [], 'area': 0, 'iscrowd': 0,
                    'image_id': idx, 'bbox': [0, 0, 0, 0], 'category_id': 0, 'id': 0
                }
            ]
        w, h = img.size
        target.append({
            'orig_size': torch.as_tensor([int(h), int(w)])
        })
        return img, target


class CocoDetectionDecorator(torchvision.datasets.CocoDetection):
    def __init__(self, original_coco_detection_dataset, dataset_wrapper, annFile: str):
        super().__init__(original_coco_detection_dataset.root, annFile)
        self.coco = original_coco_detection_dataset.coco
        self.dataset_wrapper = dataset_wrapper
        self.original_coco_detection_dataset = original_coco_detection_dataset

    def __getitem__(self, idx):
        original_img, _ = self.original_coco_detection_dataset.__getitem__(idx)
        img, target = self.dataset_wrapper.__getitem__(idx)
        w, h = original_img.size
        # Hack provide original size of image
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['image_id'] = torch.tensor([target['image_id']])
        return img, target

    def __len__(self):
        return self.original_coco_detection_dataset.__len__()


def make_coco_transforms(image_set):
    normalize = v2.Compose([
        v2.ToTensor(),
        v2.ConvertBoundingBoxFormat(format='XYXY'),
        NormalizeBoxes(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return v2.Compose([
            v2.ToImage(),
            v2.CenterCrop(512),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

    if image_set == 'val':
        return v2.Compose(
            [
                v2.ToImage(),
                v2.CenterCrop(512),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
                normalize
            ]
        )

    raise ValueError(f'unknown {image_set}')


def build(args, image_set):
    root = Path(args.dataset_folder)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set))
    wrapper_dataset = wrap_dataset_for_transforms_v2(
        dataset,
        target_keys=("boxes", "labels", 'image_id')
    )
    return CocoDetectionDecorator(
        original_coco_detection_dataset=dataset,
        dataset_wrapper=wrapper_dataset,
        annFile=ann_file
    )
