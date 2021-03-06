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

from sapir_image_captions import GLOBAL_TOKENS, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from sapir_image_captions.checkpoints import load_checkpoint
from sapir_image_captions.multi_30k.dataset import CaptionTask2Dataset
from sapir_image_captions.utils import tensor2text, remove_tokens
from sapir_image_captions.models import beam_search_caption_generation, \
    batch_beam_search_caption_generation

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str,
                        help="Directory containing model_best.pth.tar.")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing test data.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size [Default: 16].")
    parser.add_argument("--create-losses-plot", action='store_true',
                        default=True,
                        help="Create losses plot [Default: True]")
    parser.add_argument("--cuda", action='store_true', default=False,
                        help="Use cuda [Default: False].")
    parser.add_argument("--beam-size", type=int, default=10,
                        help='Beam size [Default: 10]')

    args = parser.parse_args()

    device = torch.device('cuda' \
                              if args.cuda and torch.cuda.is_available() \
                              else 'cpu')

    encoder, decoder, train_vocab, checkpoint = \
        load_checkpoint(args.model_dir, args.cuda)
    hyper_params = vars(checkpoint['cmd_line_args'])

    test_dataset = \
        CaptionTask2Dataset(args.data_dir, "test", year=hyper_params['year'],
                            caption_ext=hyper_params['language'],
                            version=hyper_params['version'],
                            image_size=hyper_params['image_size'],
                            max_seq_len=hyper_params['max_seq_len'],
                            max_size=hyper_params['max_size'])
    test_vocab = test_dataset.vocab
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

    all_recon_captions = []
    all_gold_captions = []
    pbar = tqdm.tqdm(total=len(test_loader))
    for batch_idx, batch in enumerate(test_loader):
        X_images = batch['image']
        X_captions = batch['text']
        caption_lengths = batch['text_len']
        batch_size = X_images.size(0)
        X_images = X_images.to(device)
        X_captions = X_captions.to(device)
        caption_lengths.to(device)

        first_image = X_images[0, :].unsqueeze(0)
        first_image_caption, first_image_alphas = \
            beam_search_caption_generation(first_image, encoder, decoder,
                                           train_vocab, device)

        captions, alphas = \
            batch_beam_search_caption_generation(X_images, encoder, decoder,
                                                 train_vocab, device,
                                                 k=args.beam_size)

        last_first_image = first_image
        last_first_image_caption = first_image_caption
        encoded_imgs = encoder(X_images)
        scores, captions_sorted, decode_lens, alphas, sort_idxs = \
            decoder(encoded_imgs, X_captions, caption_lengths)
        targets = captions_sorted

        scores_ = scores.view(-1, max(decode_lens),
                              scores.size(-1))
        recon_scores = torch.argmax(scores_, -1)

        # Convert to text for bleu score
        run_preprocess = \
            lambda x: remove_tokens(x, [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN])
        all_recon_captions.extend(
            [run_preprocess(sent) \
             for sent in tensor2text(recon_scores, train_vocab)])
        # Note (BP): Wrap in
        all_gold_captions.extend(
            [[run_preprocess(sent)] \
             for sent in  tensor2text(captions_sorted, test_vocab)])
        pbar.update()
    pbar.close();
    # Bleu score
    bleu_score = corpus_bleu(all_gold_captions, all_recon_captions)
    logging.info("Corpus bleu:\t{}".format(round(bleu_score, 4)))

    # Attention plot

    # Loss plot
    if args.create_losses_plot:
        losses_fp = os.path.join(args.model_dir, 'losses.csv')
        df_losses = pd.read_csv(losses_fp)
        sns.lineplot(x="epochs", y="val", hue="typ", data=df_losses)
        losses_out_fp = os.path.join(args.model_dir, "loss.png")
        logging.info("Saving losses plot to {}".format(losses_out_fp))
        plt.savefig(losses_out_fp)
