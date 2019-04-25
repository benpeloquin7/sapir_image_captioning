"""eval.py"""

import logging
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
import os
import pandas as pd
import pickle
import seaborn as sns
import torch
import tqdm

from sapir_image_captions.checkpoints import load_checkpoint
from sapir_image_captions.multi_30k.dataset import CaptionTask2Dataset
from sapir_image_captions.utils import tensor2text

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str,
                        help="Directory containing model_best.pth.tar.")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing test data.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size [Default: 64].")
    parser.add_argument("--create-losses-plot", action='store_true',
                        default=True,
                        help="Create losses plot [Default: True]")
    parser.add_argument("--cuda", action='store_true', default=False,
                        help="Use cuda [Default: False].")

    args = parser.parse_args()

    device = torch.device('cuda' \
                              if args.cuda and torch.cuda.is_available() \
                              else 'cpu')

    encoder, decoder, vocab, checkpoint = load_checkpoint(args.model_dir)
    hyper_params = vars(checkpoint['cmd_line_args'])

    test_dataset = \
        CaptionTask2Dataset(args.data_dir, "test", year=hyper_params['year'],
                            caption_ext=hyper_params['language'],
                            version=hyper_params['version'], vocab=vocab,
                            image_size=hyper_params['image_size'],
                            max_seq_len=hyper_params['max_seq_len'],
                            max_size=hyper_params['max_size'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

    pbar = tqdm.tqdm(total=len(test_loader))
    for batch_idx, batch in enumerate(test_loader):
        X_images = batch['image']
        X_captions = batch['text']
        caption_lengths = batch['text_len']
        batch_size = X_images.size(0)
        X_images = X_images.to(device)
        X_captions = X_captions.to(device)
        caption_lengths.to(device)
        encoded_imgs = encoder(X_images)
        scores, captions_sorted, decode_lens, alphas, sort_idxs = \
            decoder(encoded_imgs, X_captions, caption_lengths)
        targets = captions_sorted[:, 1:]

        scores_ = scores.view(-1, max(decode_lens),
                              scores.size(-1))
        recon_scores = torch.argmax(scores_, -1)

        # Convert to text for bleu score
        recon_captions = tensor2text(recon_scores, vocab)
        gold_captions = tensor2text(captions_sorted, vocab)
        import pdb; pdb.set_trace();


    # Bleu score


    # Attention plot

    # Loss plot
    if args.create_losses_plot:
        losses_fp = os.path.join(args.model_dir, 'losses.csv')
        df_losses = pd.read_csv(losses_fp)
        sns.lineplot(x="epochs", y="val", hue="typ", data=df_losses)
        losses_out_fp = os.path.join(args.model_dir, "loss.png")
        logging.info("Saving losses plot to {}".format(losses_out_fp))
        plt.savefig(losses_out_fp)
