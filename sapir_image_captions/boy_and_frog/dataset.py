"""dataset.py

Boy and frog story data-loader.

"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from matplotlib.colors import to_rgb


class BoyAndFrogDatasetException(Exception):
    pass


class BoyAndFrogDataset(Dataset):
    """Boy and Frog dataset.

    This book (https://www.amazon.com/Frog-Where-Are-You-Boy/dp/0803728816/ref=pd_lpo_sbs_14_img_0?_encoding=UTF8&psc=1&refRID=CRM186NYVZK5DS55WAJY)
    is referenced in Slobin (1996).

    Parameters
    ----------
    data_dir: str
        Path to directory containing image files.
    image_size: int
        Image h/w for reshaping.

    """

    def __init__(self, data_dir, image_size=256):
        super(BoyAndFrogDataset, self).__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.all_image_files = [fl for fl in os.listdir(self.data_dir) \
                                if os.path.splitext(fl)[1] == '.png']
        # Sort images into book order
        self.all_image_files = \
            sorted(self.all_image_files,
                   key=lambda x: int(os.path.splitext(x)[0][1:]))
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        image_fp = os.path.join(self.data_dir, self.all_image_files[item])
        img = Image.open(image_fp)
        img = img.convert('RGB')
        img = self.image_transform(img)
        return {'image': img}

    def __len__(self):
        return len(self.all_image_files)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Data directory.")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image h/w in pixels [Default 256].")

    args = parser.parse_args()
    dataset = BoyAndFrogDataset(args.data_dir, args.image_size)
    boy_and_frog_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)

    for batch_idx, batch in enumerate(boy_and_frog_dataloader):
        import pdb;

        pdb.set_trace()
