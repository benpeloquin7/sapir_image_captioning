"""utils.py"""

import os

import torch

from sapir_image_captions import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


class AverageMeter(object):
    """Computes and sto res the average and current value"""

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
    max_seq_length: int [Default: 50]
        Max number of tokens.
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

    original_len = len(text)
    text = [SOS_IDX] + [vocab.stoi.get(ch, UNK_IDX) for ch in text]
    text = text[:max_seq_length - 1] + [EOS_IDX]
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
    list[list[str]]
        List of tokenized captions.

    """
    res = []
    for row in torch.split(data, 1, dim=0):
        res.append([vocab.itos[idx] for idx in row.squeeze().tolist()])
    return res


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

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
