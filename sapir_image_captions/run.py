"""run.py

Assume dir structure for Multi30k data
    Captions source:
        multi30k/data
    Images source:
        multi30k/flickr30k-images

"""

import logging
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import os
import pandas as pd
import pickle
import tqdm
import seaborn as sns

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.utils import save_image

from sapir_image_captions import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, GLOBAL_TOKENS
from sapir_image_captions.checkpoints import save_checkpoint
from sapir_image_captions.models import CaptionAttentionDecoder, ImageEncoder, \
    beam_search_caption_generation, batch_beam_search_caption_generation
from sapir_image_captions.multi_30k.dataset import CaptionTask2Dataset
from sapir_image_captions.utils import AverageMeter, clip_gradient, \
    make_safe_dir, remove_tokens, save_caption, tensor2text

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None,
                        help="Data directory containing both images and "
                             "captions. Assumes directory structure"
                             "specified in run.py comments.")
    # Run params
    parser.add_argument("--n-epochs", type=int, default=120,
                        help="Number of epochs [Default: 120]")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size [Default: 32]")
    parser.add_argument("--cuda", action='store_true', default=False,
                        help="Use cuda [Default: False].")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="Run model weith debug params [Default: False].")
    parser.add_argument("--create_losses_plot", action='store_true',
                        default=True,
                        help="Create losses plot [Default: True].")
    parser.add_argument("--out-dir", type=str, default="./outputs",
                        help="Output directory [Default: ./outputs].")
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
    parser.add_argument("--fine-tune-encoder", action='store_true',
                        default=False, help="Update encoder weights "
                                            "[Default: False].")
    parser.add_argument("--encoder-lr", type=float, default=1e-4,
                        help="Encoder learning rate [Default: 1e-4].")
    parser.add_argument("--decoder-lr", type=float, default=4e-4,
                        help="Encoder learning rate [Default: 1e-4].")
    parser.add_argument("--grad-clip", type=float, default=5.,
                        help="Clip decoder gradients [Default: 5.]")
    parser.add_argument("--alpha-c", type=float, default=5.,
                        help="Regularization parameter for "
                             "'doubly stochastic attention' [Default: 1.]")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size [Default: 5].")
    # Data params
    parser.add_argument("--max-size", type=int, default=None,
                        help="Maximium number of examples [Default: None].")
    parser.add_argument("--max-seq-len", type=int, default=50,
                        help="Maximum caption sequence [Default: 50].")
    parser.add_argument("--year", type=int, default=2016,
                        help="Data year. This is particular to test datasets."
                             "[Default: 2016].")
    parser.add_argument("--language", type=str, default="en",
                        help="Captioning language [Default: 'en']")
    parser.add_argument("--version", type=int, default=1,
                        help="Data collected version. Note that there are 5"
                             "versions for the independent en/de descriptions."
                             "[Default: 1]")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image resizing [Default: 256].")
    args = parser.parse_args()

    make_safe_dir(args.out_dir)

    device = torch.device('cuda' \
                              if args.cuda and torch.cuda.is_available() \
                              else 'cpu')

    # Run mode
    if args.debug:

        args.n_epochs = 10
        args.encoded_img_size = 32
        args.attention_dim = 32
        args.embedding_dim = 16
        args.decoder_dim = 16
        args.dropout_rate = 0.
        args.max_seq_len = 50
        args.max_size = 1000
        args.image_size = 128
        debug_params = args.n_epochs, args.encoded_img_size, \
                       args.attention_dim, args.embedding_dim, \
                       args.decoder_dim, args.dropout_rate, args.max_seq_len, \
                       args.image_size, args.max_size
        logging.info("""Running in debug mode with params:
        n_epochs {}
        encoded_image_size {}
        attention_dim {}
        embedding_dim {}
        decoder_dim {}
        dropout_rate {}
        max_seq_len {}
        image_size {}
        max_size {}
        """.format(*debug_params))

    # Data
    train_dataset = \
        CaptionTask2Dataset(args.data_dir, "train", year=args.year,
                            caption_ext=args.language, version=args.version,
                            max_seq_len=args.max_seq_len,
                            image_size=args.image_size, max_size=args.max_size)
    train_vocab = train_dataset.vocab  # Use train vocab
    # Cache train vocab
    with open(os.path.join(args.out_dir, "vocab.pickle"), "wb") as fp:
        pickle.dump(train_vocab, fp)

    val_dataset = \
        CaptionTask2Dataset(args.data_dir, "val", year=args.year,
                            caption_ext=args.language, version=args.version,
                            vocab=train_vocab, max_seq_len=args.max_seq_len,
                            image_size=args.image_size, max_size=args.max_size)
    test_dataset = \
        CaptionTask2Dataset(args.data_dir, "test", year=args.year,
                            caption_ext=args.language, version=args.version,
                            vocab=train_vocab, max_seq_len=args.max_seq_len,
                            image_size=args.image_size, max_size=args.max_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

    # Models
    encoder = ImageEncoder(args.encoded_img_size)
    decoder = CaptionAttentionDecoder(args.attention_dim, args.embedding_dim,
                                      args.decoder_dim, train_vocab,
                                      dropout_rate=args.dropout_rate,
                                      device=device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Optimizers
    encoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=args.encoder_lr) if args.fine_tune_encoder else None

    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=args.decoder_lr)

    # Loss
    loss = nn.CrossEntropyLoss().to(device)

    best_loss = np.inf
    # track train, val, test, bleu, train_base, train_regularizer
    losses = np.zeros((args.n_epochs, 6))

    logging.info("Running model...")
    for epoch in range(args.n_epochs):

        # Train
        train_loss_meter_base = AverageMeter()
        train_loss_meter_regularizer = AverageMeter()
        train_loss_meter = AverageMeter()
        encoder.train()
        decoder.train()
        pbar = tqdm.tqdm(total=len(train_loader))
        for batch_idx, batch in enumerate(train_loader):
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

            # Since we decode starting with SOS_TOKEN the targets are all words
            # after SOS_TOKEN up to EOS_TOKEN
            targets = captions_sorted[:, 1:]
            scores_copy = scores.clone()
            scores, _ = \
                pack_padded_sequence(scores, decode_lens, batch_first=True)
            targets, _ = \
                pack_padded_sequence(targets, decode_lens, batch_first=True)

            base_loss = loss(scores, targets)
            # "Doubly stochastic attention regularization" from paper
            stochastic_loss = \
                args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            total_loss = base_loss + stochastic_loss

            train_loss_meter_base.update(
                base_loss.item(), batch_size)
            train_loss_meter_regularizer.update(
                stochastic_loss.item(), batch_size)
            train_loss_meter.update(
                total_loss.item(), batch_size)

            # Back prop
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()

            total_loss.backward()

            # Gradient clipping
            if args.grad_clip is not None:
                clip_gradient(decoder_optimizer, args.grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, args.grad_clip)

            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()
            pbar.update()
        pbar.close()

        # Val
        with torch.no_grad():
            val_loss_meter_base = AverageMeter()
            val_loss_meter_regularizer = AverageMeter()
            val_loss_meter = AverageMeter()
            encoder.eval()
            decoder.eval()
            pbar = tqdm.tqdm(total=len(val_loader))
            for batch_idx, batch in enumerate(val_loader):
                X_images = batch['image']
                X_captions = batch['text']
                caption_lengths = batch['text_len']
                batch_size = X_images.size(0)
                X_images = X_images.to(device)
                X_captions = X_captions.to(device)
                caption_lengths = caption_lengths.to(device)

                encoded_imgs = encoder(X_images)
                scores, captions_sorted, decode_lens, alphas, sort_idxs = \
                    decoder(encoded_imgs, X_captions, caption_lengths)
                targets = captions_sorted[:, 1:]

                scores_copy = scores.clone()
                scores, _ = \
                    pack_padded_sequence(scores, decode_lens, batch_first=True)
                targets, _ = \
                    pack_padded_sequence(targets, decode_lens, batch_first=True)

                # Cache samples on first batch
                if batch_idx == 0:
                    non_unk_constant_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]
                    preprocess = \
                        lambda x: remove_tokens(x, non_unk_constant_tokens)

                    logging.info(
                        "Caching samples for epoch {}...".format(epoch))
                    # Images
                    images_fp = \
                        os.path.join(args.out_dir,
                                     "epoch-{}.png".format(epoch))
                    # Note (BP): Remember to sort.
                    save_image(X_images[sort_idxs], images_fp)

                    # Captions
                    gold_captions = captions_sorted
                    scores_ = scores_copy.view(-1, max(decode_lens),
                                               scores_copy.size(-1))
                    recon_scores = torch.argmax(scores_, -1)
                    gold_captions_path = \
                        os.path.join(args.out_dir,
                                     "epoch-{}-gold.txt".format(epoch))
                    save_caption(gold_captions, train_vocab,
                                 gold_captions_path, preprocess=preprocess)
                    recon_captions_path = \
                        os.path.join(args.out_dir,
                                     "epoch-{}-recon.txt".format(epoch))
                    save_caption(recon_scores, train_vocab,
                                 recon_captions_path, preprocess=preprocess)
                    beam_captions_path = \
                        os.path.join(args.out_dir,
                                     "epoch-{}-beam.txt".format(epoch))
                    beam_captions, beam_alphas = \
                        batch_beam_search_caption_generation(
                            X_images[sort_idxs, :], encoder, decoder,
                            train_vocab, device, args.beam_size,
                            args.max_seq_len)
                    save_caption(beam_captions, None, beam_captions_path,
                                 preprocess=preprocess, is_words=True)
                    gold_caption_words = \
                        tensor2text(gold_captions, train_vocab)
                    gold_caption_words = \
                        [[preprocess(caption)]
                         for caption in gold_caption_words]
                    beam_captions = [preprocess(caption) for caption in beam_captions]
                    bleu_score = corpus_bleu(gold_caption_words, beam_captions)

                base_loss = loss(scores, targets)
                # "Doubly stochastic attention regularization" from paper
                stochastic_loss = \
                    args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                val_loss_meter_base = base_loss.item()
                val_loss_meter_regularizer = stochastic_loss.item()
                val_loss_meter.update(
                    val_loss_meter_base + val_loss_meter_regularizer,
                    batch_size)

                pbar.update()
            pbar.close()

        # Test
        with torch.no_grad():
            test_loss_meter_base = AverageMeter()
            test_loss_meter_regularizer = AverageMeter()
            test_loss_meter = AverageMeter()

            pbar = tqdm.tqdm(total=len(test_loader))
            for batch_idx, batch in enumerate(test_loader):
                X_images = batch['image']
                X_captions = batch['text']
                caption_lengths = batch['text_len']
                batch_size = X_images.size(0)
                X_images = X_images.to(device)
                X_captions = X_captions.to(device)
                caption_lengths = caption_lengths.to(device)

                encoded_imgs = encoder(X_images)
                scores, captions_sorted, decode_lens, alphas, sort_idxs = \
                    decoder(encoded_imgs, X_captions, caption_lengths)
                targets = captions_sorted[:, 1:]

                scores_copy = scores.clone()
                scores, _ = \
                    pack_padded_sequence(scores, decode_lens, batch_first=True)
                targets, _ = \
                    pack_padded_sequence(targets, decode_lens, batch_first=True)

                base_loss = loss(scores, targets)
                # "Doubly stochastic attention regularization" from paper
                stochastic_loss = \
                    args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                test_loss_meter_base = base_loss.item()
                test_loss_meter_regularizer = stochastic_loss.item()
                test_loss_meter.update(
                    test_loss_meter_base + test_loss_meter_regularizer,
                    batch_size)

                pbar.update()
            pbar.close()

        logging.info("Epoch {}\n train/val/test/bleu loss: {}/{}/{}/{}".format(
            epoch,
            round(train_loss_meter.avg, 3),
            round(val_loss_meter.avg, 3),
            round(test_loss_meter.avg, 3),
            round(bleu_score, 4)))

        # Log losses
        losses[epoch, 0] = train_loss_meter.avg
        losses[epoch, 1] = val_loss_meter.avg
        losses[epoch, 2] = test_loss_meter.avg
        losses[epoch, 3] = bleu_score
        losses[epoch, 4] = train_loss_meter_base.avg
        losses[epoch, 5] = train_loss_meter_regularizer.avg

        is_best = val_loss_meter.avg < best_loss
        best_loss = min(val_loss_meter.avg, best_loss)

        # Checkpoints
        state_params = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict() \
                if encoder_optimizer is not None else None,
            'decoder_state_dict': decoder.state_dict(),
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
            'train_loss': train_loss_meter.avg,
            'dev_loss': val_loss_meter.avg,
            'test_loss': test_loss_meter.avg,
            'train_loss_base': train_loss_meter_base.avg,
            'train_loss_regularizer': train_loss_meter_regularizer.avg,
            'cmd_line_args': args
        }
        _checkpoint = state_params.copy()
        save_checkpoint(_checkpoint, is_best, folder=args.out_dir)

    # Cache losses
    loss_typs = ['train', 'dev', 'test', 'bleu', 'train_base', 'train_regularizer']
    data = {
        'epochs': np.concatenate([list(range(args.n_epochs)) \
                                  for _ in
                                  range(len(loss_typs))]).tolist(),
        'typ': np.concatenate([np.repeat(typ, losses.shape[0]) \
                               for typ in loss_typs]).tolist(),
        'val': np.concatenate([losses[:, i] \
                               for i in range(losses.shape[1])]).tolist()
    }
    df_losses = pd.DataFrame(data)
    df_losses.to_csv(os.path.join(args.out_dir, "losses.csv"))

    # Losses plot
    if args.create_losses_plot:
        sns.lineplot(x="epochs", y="val", hue="typ", data=df_losses)
        losses_out_fp = os.path.join(args.out_dir, "losses.png")
        logging.info("Saving losses plot to {}".format(losses_out_fp))
        plt.savefig(losses_out_fp)
