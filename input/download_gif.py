import urllib.request
import pandas as pd
import math
from tqdm import tqdm
import multiprocessing
import os
import random

file_ = "val.txt"

df = pd.read_csv("tgif-v1.0.tsv", sep='\t')
df.columns = ["paths", "captions"]
print(df.head())


def chunk_fn(num_chunk, data):
    num_sentence = len(data)
    data_per_chunk = math.ceil(num_sentence/num_chunk)
    chunked_data = []
    for i in range(num_chunk):
        chunked_data.append(data[i*data_per_chunk: (i+1)*data_per_chunk])
    return chunked_data

def download_fn(paths):
    x = random.randint(0, 20000000000000000)
    file_path = f"{file_.split('.')[0]}_path_{x}.txt"
    f = open(file_path, "w")
    for path_ in tqdm(paths):
        caption = df[df["paths"]==path_]["captions"].tolist()[0]
        urllib.request.urlretrieve(path_, f"{file_.split('.')[0]}_data/{caption}_.gif")
        f.write(f"input/{file_.split('.')[0]}_data/{caption}_.gif\n")
    f.close()


num_chunk = 50
f = open(f"{file_.split('.')[0]}_path.txt", "w")
paths = open(file_).read().split('\n')
chunked_data = chunk_fn(num_chunk, paths)
pool = multiprocessing.Pool(processes=num_chunk)
pool.map(download_fn, chunked_data)