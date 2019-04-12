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


class CaptionDatasetException(Exception):
    pass


class CaptionTask2Dataset(Dataset):
    """Bespoke data loader *only* for task2 data.

    This should be generalized.

    """

    def __init__(self, data_dir, split, year='2016', caption_ext='en',
                 version=1, image_transform=transforms.ToTensor,
                 text_transform=None):
        super(CaptionTask2Dataset, self).__init__()
        self.data_dir = data_dir
        self.images_store = os.path.join(self.data_dir, "flickr30k-images")
        image_indices_path = os.path.join(self.data_dir,
                                          "data/task2/image_splits")
        captions_path = os.path.join(self.data_dir, "data/task2/tok")

        if caption_ext not in ['en', 'de']:
            raise CaptionDatasetException("{} is not a valid language "
                                          "tag.".format(caption_ext))

        if version not in list(range(1, 6)):
            raise CaptionDatasetException("{} is not a valid language "
                                          "version.".format(version))

        if split == 'test':
            image_indices_path = \
                os.path.join(image_indices_path,
                             "{}_{}_images.txt".format(split, year))
            captions_path = \
                os.path.join(captions_path,
                             "{}_{}.lc.norm.tok.{}.{}".format(
                                 split, year, version, caption_ext))
        elif split in ['train', 'val']:
            image_indices_path = \
                os.path.join(image_indices_path,
                             "{}_images.txt".format(split))
            captions_path = \
                os.path.join(captions_path,
                             "{}.lc.norm.tok.{}.{}".format(
                                 split, version, caption_ext))
        else:
            raise CaptionDatasetException("{} is not a valid "
                                          "split.".format(split))

        # Images
        self.image_indices = \
            pd.read_csv(image_indices_path, sep=" ", header=None) \
                .values \
                .squeeze() \
                .tolist()

        # Captions
        self.captions = pd.read_csv(captions_path, sep="\n", header=None) \
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

        caption = self.text_transform(self.captions[item])

        return {
            'image': img,
            'text': caption
        }

    def __len__(self):
        return len(self.captions)


if __name__ == '__main__':
    d = CaptionTask2Dataset('/Users/benpeloquin/Data/general/multi30k/',
                            'train', caption_ext='de')
    data = d.__getitem__(0)
