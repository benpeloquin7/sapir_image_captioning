import torch

from sapir_image_captions import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


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
