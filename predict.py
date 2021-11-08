import config
from src import model

import numpy as np
import cv2
import torch
import imageio
from transformers import BertTokenizer

def predict(gif_path):
    gif = imageio.get_reader(gif_path, ".gif")
    gif_frames = []
    for num, frame in enumerate(gif):
        if num%config.skip_frame == 0:
            frame = cv2.resize(frame, (config.image_height, config.image_height))
            if len(frame.shape) == 2:
                new_frame = np.zeros((frame.shape[0], frame.shape[1], 4))
                for i in range(4):
                    new_frame[:, :, i] = frame
                frame = new_frame

            gif_frames.append(frame/255.0)
    
    gif_frames = torch.tensor(gif_frames, dtype=torch.float)[None, ...]
    gif_frames = gif_frames.permute(0, 4, 2, 3, 1)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    
    model_ = model.GifToCaption(
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        gif_channel_out=config.gif_channel_out,
        gif_height=config.gif_height, 
        gif_width=config.gif_width,
        vocab_size=config.vocab_size, 
        embed_out=config.embed_out,
        decoder_num_layers=config.decoder_num_layers,
        forward_expansion=config.forward_expansion,
        attention_num_heads=config.attention_num_heads,
        output_max_len=config.output_max_len,
        layer_norm_ep=config.layer_norm_ep,
        encoder_dropout=config.encoder_dropout,
        decoder_dropout=config.decoder_dropout,
    )

    model_.load_state_dict(torch.load(config.MODEL_PATH))
    model_.eval()

    text = "[CLS]"
    
    with torch.no_grad():
        for _ in range(50):
            caption_idx = tokenizer.encode(text, add_special_tokens=False)
            caption_idx = torch.tensor(caption_idx, dtype=torch.long).unsqueeze(0)
            caption_out = model_(gif_frames, caption_idx)[0]
            caption_out = torch.softmax(caption_out, dim=-1)
            caption_out = torch.argmax(caption_out, dim=-1)[-1].item()
            caption_text = tokenizer.decode([caption_out])
            if caption_text == "[SEP]":
                break
            text = text + ' ' + caption_text 
    print(f"Predicted Caption: {text}")




if __name__ == "__main__":
    gif_path = "input/train_data/a man is in front of white moving triangles and he is moving his hands,_.gif"
    predict(gif_path)

    
        
    
