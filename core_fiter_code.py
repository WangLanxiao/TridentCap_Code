import numpy as np
import json
import random
import pickle as pkl
from tqdm import tqdm
from tqdm.contrib import tzip
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import transformers
import datasets
import pandas as pd
import pickle
from tqdm import tqdm
import torch
import math
import requests
import torchvision.transforms as transforms
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import os
import argparse
import time
from shutil import copyfile

def phrases(words,nn):
    phrase = []
    for word in words:
        phrase.append(word)
        if len(phrase) > nn:
            phrase.remove(phrase[0])
        if len(phrase) == nn:
            yield tuple(phrase)


# mark_score  from clip score of img and sentence
# selcted_data from the output of stage 1

low_0=[[],[],[],[],[]]
low_1=[[],[],[],[],[]]
high=[[],[],[],[],[]]
N_G=1000
for idx_data, sub in enumerate(tqdm(selcted_data)):
    words_list=sub['caption'].split(' ')
    if words_list.count('UNK')>=1:
        low_0[idx_data//N_G].append(sub)
        continue
    if any(i==j for i,j in zip(words_list, words_list[1:])):
        low_0[idx_data//N_G].append(sub)
        continue

    r1 = list(phrases(words_list, 1))
    if max(Counter(r1).values()) >= 4:
        low_0[idx_data // N_G].append(sub)
        continue

    r2=list(phrases(words_list,2))
    if max(Counter(r2).values())>=3:
        low_0[idx_data // N_G].append(sub)
        continue

    r3 = list(phrases(words_list, 3))
    if max(Counter(r3).values()) >= 2:
        low_0[idx_data // N_G].append(sub)
        continue

    r4 = list(phrases(words_list, 4))
    if max(Counter(r4).values()) >= 2:
        low_0[idx_data // N_G].append(sub)
        continue

    if mark_score[idx_data]<32.0:
        low_1[idx_data//N_G].append(sub)
        continue
    high[idx_data//N_G].append(sub)

# words filter out  ==>  low_0
# score filter out  ==>  low_1
# high quality  ==>  high
