"""models.py


Note that most of this code is modeled off the image captioning tutorial
here: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py
"""

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from sapir_image_captions import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN



class ImageEncoder(nn.Module):
    """Image encoder.

    Parameters
    ----------
    encoded_img_size: int [Default: 14]
        Fixed size to resize images.

    """
    def __init__(self, encoded_img_size=14):
        super(ImageEncoder, self).__init__()
        self.encoded_img_size = encoded_img_size

        resnet = torchvision.models.resnet101(pretrained=True)
        # Removes final linear and pool layers for our task
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize encoding to a fixed size -- while this would allow us to
        # handle varying images sizes, this isn't strictly necessary as we
        # resize all images in dataset.py
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.encoded_img_size,
                                                   self.encoded_img_size))
        self.fine_tune()

    def forward(self, images):
        """"""
        out = self.resnet(images)      # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """Allow for calculation of gradients for encoder params.

        We only fine-tune convolutional blocks 2-4 bc the first convolutional
        block contains import low-level image processing info we'd like to
        keep untouched.
        """
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """Attention network. Separate linear layers transform both the
    encoded image and the decoder hidden state (output) to the same
    size (attention_size). They are combined and passed through ReLU.
    Generate alpha by passing weights through softmax.

    Parameters
    ----------
    encoder_dim: int
        Encoded image dimension.
    decoder_dim: int
        Decoder RNN hidden dimension.
    attention_dim: int
        Attention network dimension.

    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        # Linear layer transform encoded image
        self.encoder_att = nn.Linear(self.encoder_dim, self.attention_dim)
        # Linear layer transform decoded caption
        self.decoder_att = nn.Linear(self.decoder_dim, self.attention_dim)
        # Calculate attention weights (will take softmax over these scores)
        self.full_att = nn.Linear(self.attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """TODO (BP) step through this..."""
        att_1  = self.encoder_att(encoder_out)     # (batch_size, num_pixels, attention_dim)
        att_2 = self.decoder_att(decoder_hidden)   # (batch_size, attention_dim)
        att = self.relu(att_1 + att_2.unsqueeze(1))
        attention_out = self.full_att(att).squeeze(2)
        alpha = self.softmax(attention_out)
        attention_weight_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weight_encoding, alpha


class CaptionDecoder(nn.Module):
    """Descr here...

    Parameters
    ----------
    attention_dim: int
        Attention dimension.
    embedding_dim: int
        Word embedding size.
    decoder_dim: int
        Decoder hidden RNN size.
    vocab_size: int
        Vocabulary size.
    encoder_dim: int [Default: 2048]
        Feature size of encoded images.
    dropout_rate: float [Default 0.5].
        Dropout rate.
    """

    def __init__(self, attention_dim, embedding_dim, decoder_dim, vocab,
                 encoder_dim=2048, dropout_rate=0.5,
                 device=torch.device('cpu')):
        super(CaptionDecoder, self).__init__()
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.encoder_dim = encoder_dim
        self.dropout_rate = dropout_rate
        self.device = device

        self.attention = \
            Attention(self.encoder_dim, self.decoder_dim, self.attention_dim)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.decode_step = nn.LSTMCell(self.embedding_dim + self.encoder_dim,
                                       self.decoder_dim)
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)  # Linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)  # Linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)  # Linear layer to create a sigmoid activate gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)       # Linear layer to find scores over vocab
        self.init_weights()

    def init_weights(self):
        """Initialize layers."""
        self.embedding.weight.data.uniform_(-0.1, 0.1).to(self.device)
        self.fc.bias.data.fill_(0).to(self.device)
        self.fc.weight.data.uniform_(-0.1, 0.1).to(self.device)

    def load_pretrained_embeddings(self, embeddings):
        """Load embeddings layer with pre-trained embeddings."""
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """Allow gradients to backprop to embeddings. Only use  if learning
        embeddings from scratch.
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """Create the initial hidden and cell states for the decoder's LSTM
        based on the encoded image.

        Parameters
        ----------
        encoder_out: torch.Tensor (batch_size, num_pixels, encoder_dim)
            Batch of encoded images.

        Returns
        -------
        tuple (torch.Tensor, torch.Tensor)
            Initial hidden and cell states for LSTM.

        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """

        Parameters
        ----------
        encoder_out: torch.Tensor (batch_size, enc_image_size, enc_image_size, encoder_dim)
            Image encoder output.
        encoded_captions: torch.Tensor (batch_size, max_caption_len)
            Encoded captions.
        caption_lengths: torch.Tensors (batch_size, 1)
            Caption lengths.

        Returns
        -------
        tuple (torch.Tensor, sorted encoded captions, decode lens, wieghts, sort_indices)

        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        captions_lengths, sort_idxs = \
            caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idxs]
        encoded_captions = encoded_captions[sort_idxs]

        # Embedding (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)
        # Init LSTM state (batch_size, decoder_dim)
        h, c = self.init_hidden_state(encoder_out)

        # Subtract one here since we've finished generating as soon
        # as we generate EOS_TOKEN
        decode_lengths = (captions_lengths-1).tolist()

        # Create tensors to hold word prediction scores and alphas
        # Initialize with PAD_IDX
        pad_idx = self.vocab.stoi[PAD_TOKEN]
        predictions = \
            torch.tensor([pad_idx], dtype=torch.float32) \
                .repeat((batch_size, max(decode_lengths), self.vocab_size)) \
                .to(self.device)
        alphas = \
            torch.zeros(batch_size, max(decode_lengths), num_pixels) \
                .to(self.device)

        # At each time step decode by attention-weighting the encoder's
        # output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word
        # and the attention weighted encoding
        for t in range(max(decode_lengths)):
            # Note (BP): batch_size_t handles variable sizing...
            # if we get to a point where decode lengths is less than
            # some other size between it and the max then don't update.
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = \
                self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            # gating scalar, (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            curr_embeddings = embeddings[:batch_size_t, t, :]
            curr_hidden = h[:batch_size_t]
            curr_cell = c[:batch_size_t]
            decode_in = torch.cat(
                [curr_embeddings, attention_weighted_encoding], dim=1)
            h, c = self.decode_step(decode_in, (curr_hidden, curr_cell))
            preds = self.fc(self.dropout(h))  # (batch_size, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        return predictions, encoded_captions, decode_lengths, alphas, sort_idxs


def beam_search_caption_generation(image, encoder, decoder, vocab,
                                   device, k=5, max_length=30):
    """Caption generation for a single image with beam search.

    Parameters
    ----------
    image: torch.Tensor (1, 3, h, w)
        Image.
    encoder: torch.nn.Module
        Image encoder.
    decoder: torch.nn.Module
        Caption decoder.
    vocab: torchtext.Vocab
        Vocabulary object.
    device: torch.device
        Device to run.
    k: int
        Beam size.

    Returns
    -------
    tuple (torch.tensor, torch.tensor)
        Tuple of character and alphas sequences.

    """

    encoder_out = encoder(image)
    encoded_img_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
    # Flatten encoding
    # (1, num_pixels, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)
    # We'll treat the problem as having a batch size of k
    # (k, num_pixels, encoder_dim)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
    k_prev_words = \
        torch.LongTensor([[vocab.stoi[SOS_TOKEN]]] * k).to(device)  # (k, 1)
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(device)
    seqs_alpha = \
        torch.ones(k, 1, encoded_img_size, encoded_img_size).to(device)
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)
    while True:
        # (s, embed_dim)
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        # (s, encoder_dim), (s, num_pixels)
        awe, alpha = decoder.attention(encoder_out, h)
        # (s, enc_image_size, enc_image_size)
        alpha = alpha.view(-1, encoded_img_size, encoded_img_size)
        # gating scalar, (s, encoder_dim)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe
        # (s, decoder_dim)
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / len(vocab)
        next_word_inds = top_k_words % len(vocab)

        # Add new words to sequences, alphas
        # (s, step+1)
        seqs = torch.cat([seqs[prev_word_inds],
                          next_word_inds.unsqueeze(1)], dim=1)
        # (s, step+1, enc_image_size, enc_image_size)
        seqs_alpha = \
            torch.cat([seqs_alpha[prev_word_inds],
                       alpha[prev_word_inds].unsqueeze(1)], dim=1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = \
            [ind for ind, next_word in enumerate(next_word_inds) \
             if next_word != vocab.stoi[EOS_TOKEN]]
        complete_inds = \
            list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > max_length:
            break
        step += 1

    if complete_seqs_scores == []:
        complete_seqs.extend([seqs[0].tolist()])
        complete_seqs_alpha.extend(seqs_alpha[0].tolist())
        complete_seqs_scores.extend(top_k_scores[0])
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas
