"""eval.py"""

import logging
import os

import tqdm
import torch
from torchvision.utils import save_image

from sapir_image_captions.boy_and_frog.dataset import BoyAndFrogDataset
from sapir_image_captions.checkpoints import load_checkpoint
from sapir_image_captions.models import beam_search_caption_generation
from sapir_image_captions.utils import tensor2text, make_safe_dir


logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str,
                        help="Directory containing model_best.pth.tar.")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing test data.")
    parser.add_argument("--out-dir", type=str,
                        default="boy_and_frog_eval_outputs/",
                        help="Output directory "
                             "[Default: boy_and_frog_eval_outputs]")
    args = parser.parse_args()

    encoder, decoder, train_vocab, checkpoint = \
        load_checkpoint(args.model_dir, False)

    image_size = vars(checkpoint['cmd_line_args'])["image_size"]
    dataset = BoyAndFrogDataset(args.data_dir, image_size)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)
    device = torch.device('cpu')

    make_safe_dir(args.out_dir)

    captions = []
    pbar = tqdm.tqdm(total=len(data_loader))
    for pic_idx, pic_data in enumerate(data_loader):
        image = pic_data['image']
        caption, alphas = beam_search_caption_generation(
            image, encoder, decoder, train_vocab, device, k=10)
        caption_idxs = torch.LongTensor(caption).unsqueeze(0)
        caption_words = tensor2text(caption_idxs, train_vocab)
        captions.append(caption_words)

        image_fp = \
            os.path.join(args.out_dir,
                         "img{}.png".format(pic_idx))
        save_image(image, image_fp)
        pbar.update()
    pbar.close()

    captions_fp = os.path.join(args.out_dir, "captions.txt")
    preprocess = lambda x: x
    with open(captions_fp, 'w') as fp:
        for caption in captions:
            fp.write("{}\n".format(" ".join(preprocess(caption[0]))))

    logging.info("Cacheing images and captions to {}".format(args.out_dir))
