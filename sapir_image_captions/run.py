"""run.py

Assume dir structure for Multi30k data

# Captions
multi30k/data
# Images
multi30k/flickr30k-images

"""

import logging
import tqdm

import torch
from sapir_image_captions.multi_30k.dataset import CaptionTask2Dataset

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None,
                        help="Data directory containing both images and "
                             "captions. Assumes directory structure"
                             "specified in run.py comments.")
    # Run params
    parser.add_argument("--n-epochs", type=int, default=30,
                        help="Number of epochs [Default: 30]")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size [Default: 64]")
    # Data params
    parser.add_argument("--year", type=int, default=2016,
                        help="Data year. This is particular to test datasets."
                             "[Default: 2016].")
    parser.add_argument("--language", type=str, default="en",
                        help="Captioning language [Default: 'en']")
    parser.add_argument("--version", type=int, default=1,
                        help="Data collected version. Note that there are 5"
                             "versions for the independent en/de descriptions."
                             "[Default: 1]")

    args = parser.parse_args()

    train_dataset = \
        CaptionTask2Dataset(args.data_dir, "train", year=args.year,
                            caption_ext=args.language, version=args.version)
    train_vocab = train_dataset.vocab
    val_dataset = \
        CaptionTask2Dataset(args.data_dir, "val", year=args.year,
                            caption_ext=args.language, version=args.version,
                            vocab=train_vocab)
    test_dataset = \
        CaptionTask2Dataset(args.data_dir, "test", year=args.year,
                            caption_ext=args.language, version=args.version,
                            vocab=train_vocab)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.n_epochs):

        # Train
        pbar = tqdm.tqdm(total=len(train_loader))
        for batch_idx, batch in enumerate(train_loader):
            import pdb;

            pdb.set_trace();
            pbar.update()
        pbar.close()

            # Val

            # Test
