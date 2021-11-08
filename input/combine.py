import os 
from glob import glob
import shutil
import imageio
from tqdm import tqdm

folders = ["train_data/*.gif", "val_data/*.gif"]

for folder in folders:
    all_gifs = glob(folder)
    for gif in tqdm(all_gifs):
        try:
            x = imageio.get_reader(gif, ".gif")
        except:
            os.remove(gif)