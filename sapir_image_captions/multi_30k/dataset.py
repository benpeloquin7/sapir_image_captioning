"""data.py

Get multi-30k data.

https://github.com/multi30k/dataset

Note that flickr images differ in size so we reshape below.

"""

import logging
import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from torchtext import data
from torchvision import transforms

from sapir_image_captions import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from sapir_image_captions.utils import text2tensor

logging.getLogger().setLevel(logging.INFO)


class CaptionDatasetException(Exception):
    pass


class CaptionTask2Dataset(Dataset):
    """Bespoke data loader *only* for task2 captions and images.

    Task2 includes independent descriptions.

    You may consider generalizing this.

    """

    def __init__(self, data_dir, split, year='2016', caption_ext='en',
                 version=1, min_token_freq=1, max_seq_len=50, vocab=None,
                 image_size=256, max_size=None):
        super(CaptionTask2Dataset, self).__init__()
        self.data_dir = data_dir
        self.images_store = os.path.join(self.data_dir, "flickr30k-images")
        self.split = split
        self.year = year
        self.caption_ext = caption_ext
        self.versions = range(1, 6)
        self.version = version
        self.min_token_freq = min_token_freq
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.max_size = max_size

        image_indices_path = os.path.join(self.data_dir,
                                          "data/task2/image_splits")
        captions_data_dir = os.path.join(self.data_dir, "data/task2/tok")

        if caption_ext not in ['en', 'de']:
            raise CaptionDatasetException("{} is not a valid language "
                                          "tag.".format(self.caption_ext))

        if version not in list(range(1, 6)):
            raise CaptionDatasetException("{} is not a valid language "
                                          "version.".format(self.version))

        if split == 'test':
            image_indices_path = \
                os.path.join(image_indices_path,
                             "{}_{}_images.txt".format(self.split, self.year))
            captions_paths = [
                os.path.join(captions_data_dir,
                             "{}_{}.lc.norm.tok.{}.{}".format(
                                 self.split, self.year, version,
                                 self.caption_ext)) \
                for version in self.versions]
        elif split in ['train', 'val']:
            image_indices_path = \
                os.path.join(image_indices_path,
                             "{}_images.txt".format(self.split))
            captions_paths = [
                os.path.join(captions_data_dir, "{}.lc.norm.tok.{}.{}".format(
                    self.split, version, self.caption_ext)) \
                for version in self.versions]
        else:
            raise CaptionDatasetException("{} is not a valid "
                                          "split.".format(self.split))

        # Images
        self.image_indices = \
            pd.read_csv(image_indices_path, sep=" ", header=None) \
                .values \
                .squeeze() \
                .tolist()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize
        ])
        # Captions
        self.text_field = data.Field(sequential=True, init_token=SOS_TOKEN,
                                     pad_token=PAD_TOKEN, eos_token=EOS_TOKEN,
                                     unk_token=UNK_TOKEN, lower=True,
                                     include_lengths=True, batch_first=True)
        text_field_meta = [('caption', self.text_field)]
        # Each version gets it's own dataset.
        self.captions1 = \
            data.TabularDataset(path=captions_paths[0], format='CSV',
                                fields=text_field_meta, skip_header=False,
                                csv_reader_params={'delimiter': '\n'})
        self.captions2 = \
            data.TabularDataset(path=captions_paths[1], format='CSV',
                                fields=text_field_meta, skip_header=False,
                                csv_reader_params={'delimiter': '\n'})
        self.captions3 = \
            data.TabularDataset(path=captions_paths[2], format='CSV',
                                fields=text_field_meta, skip_header=False,
                                csv_reader_params={'delimiter': '\n'})
        self.captions4 = \
            data.TabularDataset(path=captions_paths[3], format='CSV',
                                fields=text_field_meta, skip_header=False,
                                csv_reader_params={'delimiter': '\n'})
        self.captions5 = \
            data.TabularDataset(path=captions_paths[4], format='CSV',
                                fields=text_field_meta, skip_header=False,
                                csv_reader_params={'delimiter': '\n'})
        self.captions_datasets = [self.captions1,
                                  self.captions2,
                                  self.captions3,
                                  self.captions4,
                                  self.captions5]

        if self.max_size is not None:
            max_size_ = min(self.max_size, len(self.captions1.examples))
            # Reduce to max size for all caption datsets
            for caption_dataset in self.captions_datasets:
                caption_dataset.examples = \
                    caption_dataset.examples[: max_size_]
            self.image_indices = self.image_indices[: max_size_]

        if vocab is not None:
            self.vocab = vocab
        elif split == 'train' or split == 'test':
            # Note (BP): We allow test sets to write new vocab so that
            # for evalauation only.
            logging.info("Building new {} vocab...".format(split))
            self.text_field.build_vocab(self.captions_datasets[self.version-1],
                                        min_freq=self.min_token_freq)
            self.vocab = self.text_field.vocab
        else:
            # Val must have vocab
            raise CaptionDatasetException("Must pass a valid vocab"
                                          " if not train split...")

    def __getitem__(self, item):
        # Image
        image_fp = os.path.join(self.images_store, self.image_indices[item])

        img = Image.open(image_fp)
        img = self.image_transform(img)

        references = []
        for i in range(len(self.captions_datasets)):
            if i + 1 == self.version:
                caption, original_caption_len = \
                    text2tensor(self.captions_datasets[i][item].caption,
                                self.vocab, self.max_seq_len)
            # Note (BP): Important we return as string
            # here to avoid tensor-like behavior from dataloader.
            references.append(' '.join(self.captions_datasets[i][item].caption))

        return {'image': img, 'target_version': self.version,
                'captions': caption, 'caption_lens': original_caption_len,
                'references': references}

    def __len__(self):
        return len(self.captions1)


if __name__ == '__main__':
    import pickle

    d = CaptionTask2Dataset('/Users/benpeloquin/Data/general/multi30k/',
                            'train', caption_ext='en')
    vocab = d.vocab

    with open('test_vocab.pickle', 'wb') as fp:
        pickle.dump(vocab, fp)
