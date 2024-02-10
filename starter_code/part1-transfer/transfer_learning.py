# Starter code for Part 1 of the Small Data Solutions Project
# 

#Set up image data for train and test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from TrainModel import train_model
from TestModel import test_model
from torchvision import models
from pathlib import Path


# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Set up Transforms (train, val, and test)

#<<<YOUR CODE HERE>>>
train_transfroms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
)

testval_transforms = transforms.Compose(
    [
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
)


#Set up DataLoaders (train, val, and test)
batch_size = 10
num_workers = 4

#<<<YOUR CODE HERE>>>

#hint, create a variable that contains the class_names. You can get them from the ImageFolder
class_names = Path("./imagedata-50/train").glob("*/")
class_names = [Path(n).name for n in class_names]
class_names.sort()


# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifer for our 3 categories

#<<<YOUR CODE HERE>>>

# Train model with these hyperparameters
# 1. num_epochs 
# 2. criterion 
# 3. optimizer 
# 4. train_lr_scheduler 

#<<<YOUR CODE HERE>>>


# When you have all the parameters in place, uncomment these to use the functions imported above
#def main():
#   trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, train_loader, val_loader, num_epochs=num_epochs)
#   test_model(test_loader, trained_model, class_names)

#if __name__ == '__main__':
#    main()
#    print("done")