import unittest

from pytorchvideo.data import LabeledVideoDataset, ClipSampler
from torch.utils.data import RandomSampler

from ucf11 import Ucf11  # replace 'your_module' with the actual module name


class TestUcf11(unittest.TestCase):

    def test_ucf11(self):
        decode_audio = True
        decoder = "pyav"

        train_dataset, val_dataset = Ucf11(
            data_path='../UCF11_updated_mpg',
            clip_sampler=ClipSampler,
            video_sampler=RandomSampler,
            decode_audio=decode_audio,
            decoder=decoder
        )

        self.assertIsInstance(train_dataset, LabeledVideoDataset)
        self.assertIsInstance(val_dataset, LabeledVideoDataset)

        self.assertLess(val_dataset.num_videos, train_dataset.num_videos)


if __name__ == "__main__":
    unittest.main()
