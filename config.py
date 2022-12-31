import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import plotly.express as px
import os
from tqdm import tqdm
from icecream import ic

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CPUS = os.cpu_count()
BATCH_SIZE = 128