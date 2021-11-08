import warnings
warnings.filterwarnings("ignore")

import config
from src import dataloader
from src import engine
from src import model
import predict

import glob
import torch 
import transformers
import torch.nn as nn 
import albumentations

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys

def run():
    train_path = glob.glob("input/train_data/*.gif")[:10000]
    test_path = glob.glob("input/val_data/*.gif")[:2000]
    testing_gif = train_path[0]

    transforms = [
        albumentations.augmentations.transforms.Blur(always_apply=True, p=1.0),
        albumentations.augmentations.transforms.Cutout(always_apply=True, p=1.0),
        albumentations.augmentations.transforms.GaussianBlur(always_apply=True, p=1.0),
        albumentations.augmentations.transforms.RandomBrightness(always_apply=True, p=1.0),
        albumentations.augmentations.transforms.RandomContrast(always_apply=True, p=1.0),
    ]


    train_data = dataloader.DataLoader(
        gif_paths=train_path, 
        skip_frame=config.skip_frame, 
        image_height=config.image_height, 
        transforms=transforms,
        transform_threshold=config.transform_threshold
    )

    test_data = dataloader.DataLoader(
        gif_paths=test_path, 
        skip_frame=config.skip_frame, 
        image_height=config.image_height
    )

    pad_idx = train_data.tokenizer.encode("[PAD]", add_special_tokens=False)[0]

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=config.Batch_Size,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataloader.MyCollate(pad_idx, config.image_height)
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config.Batch_Size,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataloader.MyCollate(pad_idx, config.image_height),
    )

    if torch.cuda.is_available():
        accelarator = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        accelarator = 'cpu'
    
    
    device = torch.device(accelarator)

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
        device=device
    )

    model_ = model_.to(device)
    # decay_parmas = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # optimized_params = [
    #     {'params' : [p for n, p in model_.named_parameters() if not any(nd in n for nd in decay_parmas)], 'weight_decay': 0.001},
    #     {'params' : [p for n, p in model_.named_parameters() if any(nd in n for nd in decay_parmas)], 'weight_decay': 0.0}
    # ]


    optimizer = transformers.AdamW(model_.parameters(), lr=config.LR)
    num_training_steps = config.Epochs*len(train_data)//config.Batch_Size
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.Warmup_steps*num_training_steps,
        num_training_steps=num_training_steps
    )
    # scheduler = None

    best_loss = 1e4
    best_model = model_.state_dict()
    print('--------- [INFO] STARTING TRAINING ---------')
    for epoch in range(config.Epochs):
        train_loss = engine.train_fn(model_, train_loader, optimizer, scheduler, device)
        val_loss = engine.eval_fn(model_, test_loader, device)
        print(f'EPOCH -> {epoch+1}/{config.Epochs} | TRAIN LOSS = {train_loss} | VAL LOSS = {val_loss}')
        print(testing_gif)
        predict.predict(testing_gif)
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model_.state_dict()
            torch.save(best_model, config.MODEL_PATH)

if __name__ == "__main__":
    run()