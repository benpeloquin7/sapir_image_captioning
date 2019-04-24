"""checkpoints.py"""

import logging
import os
import pickle
import shutil

import torch

from sapir_image_captions.models import ImageEncoder, CaptionDecoder

logging.getLogger().setLevel(logging.INFO)


def save_checkpoint(state, is_best, folder='./',
                    filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(dir_path, use_cuda=False):
    """Load model checkpoint.

    Parameters
    ----------
    dir_path: str
        Path to directory containing.
            + model_best.pth.tar
            + vocab.pickle
    use_cuda: bool [Default: False]
        Use cuda flag.

    Returns
    -------
    tuple
        encoder model, decoder model, vocab, checkpoin

    """
    model_path = os.path.join(dir_path, "model_best.pth.tar")
    vocab_path = os.path.join(dir_path, "vocab.pickle")
    with open(vocab_path, 'rb') as fp:
        vocab = pickle.load(fp)



    checkpoint = torch.load(model_path) if use_cuda else \
        torch.load(model_path, map_location=lambda storage, location: storage)
    args = vars(checkpoint['cmd_line_args'])
    logging.info("""
    Loading best model -->
        Epoch:\t{}
        train loss:\t{}\n
        dev loss:\t{}\n
        test loss:\t{}\n
    """.format(checkpoint['epoch'],
               checkpoint['train_loss'],
               checkpoint['dev_loss'],
               checkpoint['test_loss']))

    device = torch.device('cuda' \
                              if args['cuda']and torch.cuda.is_available() \
                              else 'cpu')

    encoder = ImageEncoder(args['encoded_img_size'])
    decoder = CaptionDecoder(args['attention_dim'], args['embedding_dim'],
                             args['decoder_dim'], len(vocab),
                             dropout_rate=args['dropout_rate'], device=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return encoder, decoder, vocab, checkpoint
