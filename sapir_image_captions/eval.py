"""eval.py"""

import logging
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
import os
import pandas as pd
import pickle
import seaborn as sns

from sapir_image_captions.checkpoints import load_checkpoint
from sapir_image_captions.multi_30k.dataset import CaptionTask2Dataset

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str,
                        help="Directory containing model_best.pth.tar.")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing test data.")
    parser.add_argument("--create-losses-plot", action='store_true',
                        default=True,
                        help="Create losses plot [Default: True]")

    args = parser.parse_args()

    checkpoint = load_checkpoint(args.model_dir)
    import pdb; pdb.set_trace();

    with open(os.path.join(args.model_dir, "vocab.pickle")) as fp:
        vocab = pickle.load(fp)

    test_dataset = \
        CaptionTask2Dataset(args.data_dir, "test", year=args.year,
                            caption_ext=args.language, version=args.version,
                            vocab=vocab, max_seq_len=args.max_seq_len,
                            image_size=args.image_size, max_size=args.max_size)
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
