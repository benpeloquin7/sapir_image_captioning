"""model_as_participant.py

Functionality for "running" IC models producing

1. alphas saliency maps
2. image + language embeddings (not implemented)

"""

import logging
import numpy as np
import os
import pickle

import torch
import tqdm

from sapir_image_captions import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from sapir_image_captions.checkpoints import load_checkpoint
from sapir_image_captions.models import batch_beam_search_caption_generation
from sapir_image_captions.multi_30k.dataset import CaptionTask2Dataset
from sapir_image_captions.utils import AverageMeter, tensor2text, \
    remove_tokens, make_safe_dir

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir_path", type=str,
                        help="Path to model dir (containing model and "
                             "vocab files.")
    parser.add_argument("data_dir", type=str,
                        help="Path to test data.")
    parser.add_argument("--beam-size", type=int, default=10,
                        help="Beam size [Default: 10].")
    parser.add_argument("--version", type=int, default=1,
                        help="Data version [Default: 1].")
    parser.add_argument("--year", type=int, default=2016,
                        help="Data year [Default: 2016].")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size [Default: 64].")
    parser.add_argument("--max-size", type=int, default=None,
                        help="Max limit on num examples [Default: None].")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="Use cuda [Default: False].")
    parser.add_argument("--out-dir", type=str,
                        default="model_as_participant_out",
                        help="Output directory "
                             "[Default: model_as_participant_out]")

    args = parser.parse_args()



    encoder, decoder, vocab, checkpoint = load_checkpoint(args.model_dir_path)
    hyper_params = vars(checkpoint['cmd_line_args'])

    device = torch.device('cuda' \
                              if args.cuda and torch.cuda.is_available() \
                              else 'cpu')

    test_dataset = \
        CaptionTask2Dataset(args.data_dir, "test", year=args.year,
                            caption_ext=hyper_params['language'],
                            version=args.version, vocab=vocab,
                            max_seq_len=hyper_params['max_seq_len'],
                            image_size=hyper_params['image_size'],
                            max_size=args.max_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    make_safe_dir(args.out_dir)

    recon_captions_hold = []
    recon_caption_lens_hold = []
    target_captions_hold = []
    target_caption_lens_hold = []
    alphas_hold = []
    images_hold = []

    with torch.no_grad():
        val_loss_meter_base = AverageMeter()
        val_loss_meter_regularizer = AverageMeter()
        val_loss_meter = AverageMeter()
        encoder.eval()
        decoder.eval()
        pbar = tqdm.tqdm(total=len(test_loader))

        for batch_idx, batch in enumerate(test_loader):
            X_images = batch['image']
            target_version = batch['target_version'].unique().item()
            X_captions = batch['captions']
            caption_lengths = batch['caption_lens']
            batch_size = X_images.size(0)

            X_images = X_images.to(device)
            X_captions = X_captions.to(device)
            caption_lengths = caption_lengths.to(device)

            images_hold.append(X_images.numpy())

            # Run beam-search
            recon_captions, alphas = batch_beam_search_caption_generation(
                X_images, encoder, decoder, vocab, device,
                hyper_params["max_seq_len"])

            curr_target_captions = []
            curr_target_caption_lens = []
            curr_recon_captions = []
            curr_recon_caption_lens = []

            run_preprocess = \
                lambda x: remove_tokens(x, [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN])

            for target_caption, recon_caption in zip(
                    torch.split(X_captions, 1, 0), recon_captions):
                # Targets
                target = run_preprocess(tensor2text(target_caption, vocab)[0])
                curr_target_captions.append(" ".join(target))
                curr_target_caption_lens.append(len(target))

                # Reconstructions
                recon = run_preprocess(recon_caption)
                curr_recon_captions.append(" ".join(recon))
                curr_recon_caption_lens.append(len(recon))

            target_captions_hold.append(curr_target_captions)
            target_caption_lens_hold.append(curr_target_caption_lens)
            recon_captions_hold.append(curr_recon_captions)
            recon_caption_lens_hold.append(curr_recon_caption_lens)
            # A little hacky, but SOS and EOS are included in alphas so add
            # two here
            len_idxs = 2 * np.ones(len(curr_recon_caption_lens), dtype=int) \
                       + np.array(curr_recon_caption_lens)
            # Last bunch is empty vector
            alphas_split = np.split(alphas, np.cumsum(len_idxs))
            assert len(alphas_split[-1]) == 0
            # Remove SOS and EOS alphas
            alphas_split = [a[1:-1] for a in alphas_split[:-1]]
            alphas_hold.append(alphas_split)
            pbar.update()
        pbar.close()

    data = {
        "num_batches": np.ceil(len(test_dataset) / args.batch_size),
        "language": hyper_params["language"],
        "version": hyper_params["version"],
        "year": hyper_params["year"],
        "images": images_hold,
        "target_captions": target_captions_hold,
        "target_caption_lens": target_caption_lens_hold,
        "recon_captions": recon_captions_hold,
        "recon_caption_lens": recon_caption_lens_hold,
        "alphas": alphas_hold
    }

    model_fname = "model_data_{}_{}_{}.pickle".format(
        hyper_params["language"],
        hyper_params["version"],
        hyper_params["year"])
    out_path = os.path.join(args.out_dir, model_fname)
    logging.info("Logging data to {}".format(out_path))
    with open(out_path, "wb") as fp:
        pickle.dump(data, fp)
