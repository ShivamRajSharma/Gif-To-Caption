import torch
import torch.nn as nn
from tqdm import tqdm

def loss_fn(target, output):
    return nn.CrossEntropyLoss()(output, target)


def train_fn(model, dataloader, optimizer, scheduler, device):
    running_loss = 0
    model.train()
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None
        gifs = data['gif'].to(device)
        captions = data['caption_idx'].to(device)
        captions_input = captions[:, :-1]
        captions_target = captions[:, 1:]
        output = model(gifs, captions_input)
        loss = loss_fn(
            captions_target.reshape(-1), 
            output.view(-1, output.shape[2])
            )
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    epoch_loss = running_loss/len(dataloader)

    return  epoch_loss

def eval_fn(model, dataloader, device):
    running_loss = 0
    running_acc = 0
    model.eval()
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            gifs = data['gif'].to(device)
            captions = data['caption_idx'].to(device)
            captions_input = captions[:, :-1]
            captions_target = captions[:, 1:]
            output = model(gifs, captions_input)
            loss = loss_fn(
                captions_target.reshape(-1), 
                output.view(-1, output.shape[2])
                )
            running_loss += loss.item()

    epoch_loss = running_loss/len(dataloader)
    
    return  epoch_loss