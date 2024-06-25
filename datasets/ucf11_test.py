import sys
import unittest

import torchvision
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from pytorchvideo.transforms import ShortSideScale, Normalize, UniformCropVideo, ApplyTransformToKey, \
    UniformTemporalSubsample
from torch.utils.data import RandomSampler
from torchvision.transforms import functional

from ucf11 import Ucf11  # replace 'your_module' with the actual module name


class TestUcf11(unittest.TestCase):

    def test_ucf11(self):
        decode_audio = True
        decoder = "pyav"
        clip_duration = 2.0
        train_dataset, val_dataset = Ucf11(
            data_path='../UCF11_updated_mpg',
            clip_sampler=make_clip_sampler('random', clip_duration),
            video_sampler=RandomSampler,
            decode_audio=decode_audio,
            decoder=decoder,
            transform=torchvision.transforms.Compose(
                [
                    ApplyTransformToKey(
                        key="video",
                        transform=torchvision.transforms.Compose([
                            ShortSideScale(size=224),
                            UniformTemporalSubsample(num_samples=60),
                            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])),
                    UniformCropVideo(size=224),
                ]
            ))

        self.assertIsInstance(train_dataset, LabeledVideoDataset)
        self.assertIsInstance(val_dataset, LabeledVideoDataset)

        self.assertLess(val_dataset.num_videos, train_dataset.num_videos)
        correct_shape = (3, 60, 224, 224)
        self.assertTrue(all(video['video'].shape == correct_shape for video in val_dataset))



if __name__ == "__main__":
    sys.modules["torchvision.transforms.functional_tensor"] = functional
    unittest.main()
