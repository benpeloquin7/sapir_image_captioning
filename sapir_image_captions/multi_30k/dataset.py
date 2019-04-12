"""data.py

Get multi-30k data.

https://github.com/multi30k/dataset

TODO: Looks like differences in images sizes between train/val/test?

[3, 333, 500] vs [3, XXX, 500]???

"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from sapir_image_captions.multi_30k import LANGUAGE_TAGS


class CaptionDatasetException(Exception):
    pass


class CaptionTask2Dataset(Dataset):
    """Bespoke data loader *only* for task2 data.

    This should be generalized.

    """

    def __init__(self, data_dir, split, version='2016',
                 image_transform=transforms.ToTensor, text_transform=None):
        super(CaptionTask2Dataset, self).__init__()
        self.data_dir = data_dir
        self.images_store = os.path.join(self.data_dir, "flickr30k-images")
        indices_path = os.path.join(self.data_dir, "data/task2/image_splits")
        if split == 'test':
            image_indices_path = \
                os.path.join(indices_path,
                             "{}_{}_images.txt".format(split, version))
        elif split in ['train', 'val']:
            image_indices_path = \
                os.path.join(indices_path,
                             "{}_images.txt".format(split))
        else:
            raise CaptionDatasetException("{} is not a valid "
                                          "split.".format(split))
        # Image files from split.
        self.image_indices = \
            pd.read_csv(image_indices_path, sep=" ", header=None) \
                .values \
                .squeeze() \
                .tolist()

        # Set transforms
        if image_transform is not None:
            self.image_transform = image_transform()
        else:
            self.image_transform = lambda x: x

        if text_transform is not None:
            self.text_transform = text_transform()
        else:
            self.text_transform = lambda x: x

    def __getitem__(self, item):
        # Image
        image_fp = os.path.join(self.images_store, self.image_indices[item])
        img = Image.open(image_fp)
        img = self.image_transform(img)

        # Text
        text = None

        return {
            'image': img,
            'text': text
        }

    def __len__(self):
        pass


if __name__ == '__main__':
    d = CaptionTask2Dataset('/Users/benpeloquin/Data/general/multi30k/',
                            'val')
    data = d.__getitem__(0)
    import pdb; pdb.set_trace();

