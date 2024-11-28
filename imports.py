import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import kagglehub
from tqdm import tqdm

import torchvision.transforms.v2 as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets as datasets
from torchvision.models import resnet50, ResNet50_Weights

from yaml_reader import yaml_reader


config = yaml_reader()

test_path = config["dataset_parameters"]["test_path"]
train_path = config["dataset_parameters"]["train_path"]

classlist = os.listdir(train_path)
num_classes = len(classlist)


img_row = config["training_parameters"]["img_row"]
img_cols = config["training_parameters"]["img_cols"]
lr = config["training_parameters"]["lr"]
epochs = config["training_parameters"]["epochs"]
batch_size = config["training_parameters"]["batch_size"]
dropout_rate = config["model_parameters"]["dropout_rate"]
weight_decay = config["model_parameters"]["weight_decay"]

train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []
val_precision = []
val_recall = []
val_precision_per_epoch = []
val_recall_per_epoch = []
lr_list = []
best_loss = None


test_loss = []
test_accuracy = []
test_precision = []
test_recall = []
test_precision_per_epoch = []
test_recall_per_epoch = []


output_model_dir = config["output_parameters"]["out_model_directory"]
output_graphics_dir = config["output_parameters"]["out_graphics_directory"]
output_inference_dir = config["output_parameters"]["out_inference_directory"]


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
