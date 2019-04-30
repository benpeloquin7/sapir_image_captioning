"""utils.py"""

import matplotlib.pyplot as plt
import os

import torch
from torchvision.utils import make_grid, save_image

from sapir_image_captions import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

GLOBAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def text2tensor(text, vocab, max_seq_length=30, device='cpu'):
    """Convert string text into numerical tensor.

    Parameters
    ----------
    text: list[str]
        List of tuples of strs
    vocab: torchtext.data.Vocab object
        Vocab contains itos and stoi.
    max_seq_length: int [Default: 30]
        Max number of tokens (including SOS and EOS).
    device: str [Default: 'cpu']
        torch.device string. One of cpu/cuda.

    Returns
    -------
    torch.LongTensor (batch_size, max_seq_len)
        Captions tensor.

    """
    UNK_IDX, SOS_IDX, EOS_IDX, PAD_IDX = \
        map(lambda x: vocab.stoi[x],
            [UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN])
    # Set length to original + 2 (EOS and SOS) or max if original is larger
    # than max
    original_len = min(len(text)+2, max_seq_length)
    # Add SOS and EOS
    text = [SOS_IDX] + [vocab.stoi.get(ch, UNK_IDX) for ch in text]
    text = text[:max_seq_length - 1] + [EOS_IDX]
    # Padding
    text = text + [PAD_IDX] * (max_seq_length - len(text))
    return torch.LongTensor(text, device=device), original_len


def tensor2text(data, vocab):
    """Convert a tensor to text.

    Parameters
    ----------
    t: torch.Tensor (batch-size, max_seq_len)
        Batch captions.
    vocab: torchtext.data.Vocab object
        Vocab contains itos and stoi.

    Returns
    -------
    list[list[str]]
        List of tokenized captions.

    """
    res = []
    for row in torch.split(data, 1, dim=0):
        res.append([vocab.itos[idx] for idx in row.squeeze().tolist()])
    return res


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid
    explosion of gradients.

    Parameters
    -----------
    optimizer: torch.optim
        Optimizer with the gradients to be clipped
    grad_clip: float
        Clip value

    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def make_safe_dir(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


def images2grid(image_batch):
    batch_size, c, h, w = image_batch.size()
    nrows = int(batch_size / 2)
    grid_img = make_grid(image_batch, nrow=nrows)
    return grid_img


def save_caption(caption_batch, vocab, f_path, preprocess=lambda x: x):
    captions = tensor2text(caption_batch, vocab)
    with open(f_path, 'w') as fp:
        for caption in captions:
            fp.write("{}\n".format(" ".join(preprocess(caption))))


def remove_tokens(sent, bad_token_list):
    """Preprocessing functinoality for captions.

        Parameters
        ----------
        sent: list[str]
            List of string tokens.

        Returns
        -------
        list[str]
            List of string tokens with SOS and EOS removed.

    """
    return [tok for tok in sent if tok not in bad_token_list]


def remove_eos_sos(sent):
    """Preprocessing functinoality for captions.

    Parameters
    ----------
    sent: list[str]
        List of string tokens.

    Returns
    -------
    list[str]
        List of string tokens with SOS and EOS removed.

    """
    return remove_tokens(sent, [EOS_TOKEN, SOS_TOKEN])
