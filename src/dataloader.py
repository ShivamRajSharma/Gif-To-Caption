from transformers import BertTokenizer
import torch 
import imageio
import cv2
import random
import numpy as np

import os

class DataLoader(torch.utils.data.Dataset):
    def __init__(
        self, 
        gif_paths, 
        skip_frame=2, 
        image_height=64, 
        transforms=None,
        transform_threshold=0.0,
    ):
        self.gif_paths = gif_paths
        self.transforms = transforms
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.skip_frame = skip_frame
        self.image_height = image_height
        self.transform_threshold = transform_threshold
    
    def __len__(self):
        return len(self.gif_paths)
    
    def __getitem__(self, idx):
        gif_path = self.gif_paths[idx]
        caption = gif_path.split("/")[-1].split("_")[0]
        caption_idx = self.tokenizer.encode(caption)
        gif = imageio.get_reader(gif_path, ".gif")
        gif_frames = []
        if self.transforms is not None:
            random_augmentation = self.transforms[random.randint(0, len(self.transforms)-1)]
        a = random.randint(0, 100)
        for num, frame in enumerate(gif):
            if num%self.skip_frame == 0:
                frame = cv2.resize(frame, (self.image_height, self.image_height))
                if (self.transforms is not None) and (a > self.transform_threshold*100):
                    frame = random_augmentation(image=frame)["image"]
                if len(frame.shape) == 2:
                    new_frame = np.zeros((frame.shape[0], frame.shape[1], 4))
                    for i in range(4):
                        new_frame[:, :, i] = frame
                    frame = new_frame

                gif_frames.append(frame/255.0)

        return {
            "gif": torch.tensor(gif_frames, dtype=torch.float),
            "caption_idx": torch.tensor(caption_idx, dtype=torch.long),
            "caption": caption
        }


class MyCollate:
    def __init__(self, pad_idx, image_height):
        self.pad_idx = pad_idx
        self.image_height = image_height
    
    def __call__(self, batch):
        gif = [item["gif"] for item in batch]
        caption_idx = [item["caption_idx"] for item in batch]
        caption = [item["caption"] for item in batch]

        padded_caption_idx = torch.nn.utils.rnn.pad_sequence(
            caption_idx,
            batch_first=True,
            padding_value=self.pad_idx
        )

        max_len = max([x.shape[0] for x in gif])
        final_padded_gif = []
        for x in gif:
            padded_gif = torch.zeros(max_len, self.image_height, self.image_height, 4)
            x_len = x.shape[0]
            padded_gif[:x_len] = x
            final_padded_gif.append(padded_gif)
        
        final_padded_gif = torch.stack(final_padded_gif).permute(0, 4, 2, 3, 1)

        return {
            "gif": final_padded_gif,
            "caption_idx": padded_caption_idx,
            "caption": caption
        }