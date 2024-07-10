# MNIST and Dog-Cat Classification with PyTorch

This repository contains two Jupyter notebooks focusing on different image classification tasks using PyTorch. The first notebook deals with the MNIST dataset, while the second notebook focuses on the Dog-Cat classification using transfer learning with weights obtained from MNIST.

## Project Overview

1. [MNIST Classification]
    - [Loading the Dataset]
    - [Model Development]
    - [Training the Models]
    - [Fine-Tuning Saved Models]
    - [K-Fold Cross-Validation]
2. [Dog-Cat Classification]
    - [Data Loading]
    - [Fine-Tuning Saved Models]
    - [Transfer Learning with MNIST Weights]

## MNIST Classification

### Model Development

We develop three types of neural networks for classification:

1. **Fully Connected Network**
2. **Hybrid Network**
3. **Fully Convolutional Network**


### Loading the Dataset
We use PyTorch to load the MNIST dataset and create a DataLoader to handle batching and shuffling.

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor                            
from torch.utils.data import DataLoader, random_split

# Load dataset
train_val_data = MNIST(
    root = "data",                                                          
    train=True,                                                              
    transform=ToTensor(),                                                   
    target_transform=None,                                                  
    download=False
)

test_data = MNIST(
    root = "data",
    train = False,
    transform=ToTensor(),
    target_transform=None,
    download=False
)

# Split dataset
from torch.utils.data import random_split
train_size = int(0.8 * len(train_val_data))
val_size = len(train_val_data) - train_size
train_data, val_data = random_split(train_val_data, [train_size, val_size])

# Create DataLoader
train_dataloader = DataLoader(train_data, 64, True)
val_dataloader = DataLoader(val_data, shuffle=True)
test_dataloader = DataLoader(test_data, shuffle=True)
```


## Datasets

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) from PyTorch
- [Dog vs Cat Dataset](https://www.kaggle.com/c/dogs-vs-cats) from Kaggle


## Clone the repository:

```
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
```
pip install -r requirements.txt
    ```



