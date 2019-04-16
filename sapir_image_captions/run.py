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
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from sapir_image_captions.models import CaptionDecoder, ImageEncoder
from sapir_image_captions.multi_30k.dataset import CaptionTask2Dataset
from sapir_image_captions.utils import text2tensor, tensor2text

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
    parser.add_argument("--cuda", action='store_true', default=False,
                        help="Use cuda [Default: False].")
    # Model params
    parser.add_argument("--encoded-img-size", type=int, default=14,
                        help="Encoded image size [Default: 14].")
    parser.add_argument("--attention-dim", type=int, default=512,
                        help="Attention dims [Default: 512].")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Embedding dims [Default: 256].")
    parser.add_argument("--decoder-dim", type=int, default=256,
                        help="Decoder hidden dims [Default: 256].")
    parser.add_argument("--dropout-rate", type=float, default=0.5,
                        help="Dropout rate [Default: 0.5].")
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


    device = 'cuda' if args.cuda else 'cpu'

    # Data
    train_dataset = \
        CaptionTask2Dataset(args.data_dir, "train", year=args.year,
                            caption_ext=args.language, version=args.version)
    train_vocab = train_dataset.vocab  # Use train vocab
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

    # Models
    encoder = ImageEncoder(args.encoded_img_size)
    decoder = CaptionDecoder(args.attention_dim, args.embedding_dim,
                             args.decoder_dim, len(train_vocab),
                             dropout_rate=args.dropout_rate)

    loss = nn.CrossEntropyLoss().to()

    for epoch in range(args.n_epochs):

        # Train
        pbar = tqdm.tqdm(total=len(train_loader))
        for batch_idx, batch in enumerate(train_loader):
            # import pdb; pdb.set_trace();
            X_images = batch['image']
            X_captions = batch['text']
            caption_lengths = batch['text_len']

            encoded_imgs = encoder(X_images)
            scores, captions_sorted, decode_lens, alphas, sort_idxs = \
                decoder(encoded_imgs, X_captions, caption_lengths)
            targets = captions_sorted[:, 1:]

            scores_copy = scores.clone()
            scores, _ = \
                pack_padded_sequence(scores, decode_lens, batch_first=True)
            targets, _ = \
                pack_padded_sequence(targets, decode_lens, batch_first=True)

        pbar.close()

        # Val
        pbar = tqdm.tqdm(total=len(val_loader))
        for batch_idx, batch in enumerate(val_loader):
            pbar.update()
        pbar.close()

        # Test
        # Val
        pbar = tqdm.tqdm(total=len(test_loader))
        for batch_idx, batch in enumerate(test_loader):
            pbar.update()
        pbar.close()
