# Imports and check for GPU availability
import torch
from torch import *
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
import matplotlib
from torchvision import datasets
import torch.utils.model_zoo as model_zoo
from torch.utils.data.sampler import SubsetRandomSampler
import random as r
import cv2
import time

# Import utils functions
from utils import *

if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
    print("running on GPU")
else:
    device = torch.device("cpu")
    use_cuda = False
    print("running on CPU")

lr_d = 1
lr_c = 1
if use_cuda:
    model_c = Completion().cuda()
    model_d = Discriminator().cuda()
else:
    model_c = Completion()
    model_d = Discriminator()
opt_d = torch.optim.Adadelta(model_d.parameters(), lr=lr_d)
opt_c = torch.optim.Adadelta(model_c.parameters(), lr=lr_c)

transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((256, 256), scale=(0.6666666666667, 1.0), ratio=(1.0, 1.0)),
        transforms.ToTensor(),
    ]
)
train_set = torchvision.datasets.ImageFolder(root="data/celeba_train", transform=transform)
test_set = torchvision.datasets.ImageFolder(root="./data/celeba_test", transform=transform)
dataset_with_labels = True
test_loader = torch.utils.data.DataLoader(test_set)

# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# COMPLETION NETWORK LOSS 1
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
def C_loss(input, output, mask):
    """
    Computes the loss for the completion network when trained on its own (calculated only on the completed region)
    * inputs :
      - input : the original image
      - output : the completed image
      - mask : the mask that can be seen as the hole in the image
    """
    return F.mse_loss(output * mask, input * mask)


# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# COMPLETION NETWORK TRAINING ALGORITHM
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
def train_C(
    model_c,
    optimizer=opt_c,
    train_set=train_set,
    learning_rate=1.0,
    train_acc_period=100,
    n_epoch=5,
    save_period=10000,
    batch_size=8,
    num_samples=1000,
    use_cuda=True,
    pixel=(0, 0, 0),
    dataset_with_labels=True,
):
    """
    Algorithm that trains the completion network on its own
    * inputs :
      - model_c : the completion network
      - optimizer : the optimizer for the completion network
      - train_set : training dataset
      - learning_rate : learning rate
      - train_acc_period : intervals at which a loss is printed
      - n_epochs : number of epochs
      - save_period : intervals at which the model is automatically saved
      - batch_size : training batch size
      - num_samples : number of samples in the DataLoader sampler
      - use_cuda : boolean indicating the use of a GPU
      - pixel : the mean pixel on the training dataset
      - dataset_with_labels : boolean, True is the dataset used has labels
    """
    print("Beginning the training of the Completion Network")
    model_c.train()
    i = 0
    for epoch in tqdm(range(n_epoch)):
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=6,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=num_samples),
            pin_memory=True,
            drop_last=True,
        )  # On en parlera de ces paramètres là !
        running_loss = 0.0
        for data in tqdm(train_loader):
            if dataset_with_labels:
                x, y = data
            else:
                x = data

            if use_cuda:
                x = x.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            m, _ = mask(batch_size, use_cuda=use_cuda)
            inputs = apply_mask(x, m, pixel, use_cuda=use_cuda)
            outputs = model_c(inputs)

            loss = C_loss(x, outputs, m)
            loss.backward()
            optimizer.step()

            # printing some statistics
            running_loss = 0.33 * loss.item() / batch_size + 0.66 * running_loss
            if i % train_acc_period == train_acc_period - 1:
                print("[%d, %5d] loss: %f" % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
            i += 1
            del x  # Je sais pas si c'est utile ! mon but c'est de désalouer la mémoire pour être sûr de pas dépasser la capacité de CUDA
        if (epoch % save_period == save_period - 1) or (epoch == n_epoch - 1):
            filename = "model_c_save/model_c_checkpoint_c_training_epoch{:d}.pth.tar".format(epoch + 1)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_c.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                filename=filename,
            )
    print("Finished Training C")


# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# DISCRIMINATION NETWORK TRAINING ALGORITHM
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
def train_D(
    model_d,
    model_c,
    optimizer=opt_d,
    train_set=train_set,
    learning_rate=1.0,
    train_acc_period=100,
    n_epoch=5,
    save_period=10000,
    batch_size=8,
    num_samples=1000,
    use_cuda=True,
    pixel=(0, 0, 0),
    dataset_with_labels=True,
):
    """
    Algorithm that trains the discrimination network on its own
    * inputs :
      - model_d : the discrimination network
      - optimizer : the optimizer for the discrimination network
      - train_set : training dataset
      - learning_rate : learning rate
      - train_acc_period : intervals at which a loss is printed
      - n_epochs : number of epochs
      - save_period : intervals at which the model is automatically saved
      - batch_size : training batch size
      - num_samples : number of samples in the DataLoader sampler
      - use_cuda : boolean indicating the use of a GPU
      - pixel : the mean pixel on the training dataset
      - dataset_with_labels : boolean, True is the dataset used has labels
    """
    print("Beginning the training of the Discrimination network")
    model_c.eval()
    model_d.train()
    i = 0
    criterion = nn.BCELoss()
    for epoch in tqdm(range(n_epoch)):
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=6,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=num_samples),
            pin_memory=True,
            drop_last=True,
        )  # On en parlera de ces paramètres là !
        running_loss = 0.0
        for data in tqdm(train_loader):
            if dataset_with_labels:
                x, _ = data
            else:
                x = data

            if use_cuda:
                x = x.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # Feeding the network with completed images

            mask_c, centers_c = mask(batch_size, use_cuda=use_cuda)
            input_c = apply_mask(x, mask_c, pixel, use_cuda=use_cuda)
            output_c = model_c(input_c)
            false = torch.zeros(
                (batch_size, 1)
            )  # The context discriminator should label all images as fake
            if use_cuda:
                false = false.cuda()
            input_gd_fake = (
                output_c.detach()
            )  # We don't need the gradients for this batch, as the Generator is not trained here
            input_ld_fake = hole_cropping(input_gd_fake, centers_c, use_cuda=use_cuda)
            output_fake = model_d((input_ld_fake, input_gd_fake))
            loss_fake = criterion(output_fake, false)  # Loss if labelled as true

            # Feeding the network with real images
            centers_d = mask(
                batch_size, use_cuda=use_cuda, generate_mask=False
            )  # Only generate the hole centers, but not the tensors as it is expensive !
            input_gd_real = x
            input_ld_real = hole_cropping(input_gd_real, centers_d, use_cuda=use_cuda)
            output_real = model_d((input_ld_real, input_gd_real))
            real = torch.ones(
                (batch_size, 1)
            )  # Tensor of ones indicating that images are labelled as true (1)
            if use_cuda:
                real = real.cuda()
            loss_real = criterion(output_real, real)  # Loss if labelled as false

            # Recombining the losses
            loss = (
                loss_real + loss_fake
            ) / 2.0  # The coefficient of each kind of losses could be a way to personalize the net !
            loss.backward()
            optimizer.step()

            # printing some statistics
            running_loss = 0.33 * loss.item() / batch_size + 0.66 * running_loss
            if i % train_acc_period == train_acc_period - 1:
                print("[%d, %5d] loss: %f" % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
            i += 1
            del x
        if (epoch % save_period == save_period - 1) or (epoch == n_epoch - 1):
            filename = "model_d_save/model_d_checkpoint_d_training_epoch{:d}.pth.tar".format(epoch + 1)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_d.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                filename=filename,
            )
    print("Finished Training D")


# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# ADVERSARIAL TRAINING ALGORITHM
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
def train_C_and_D(
    model_c,
    model_d,
    alpha,
    optimizer_c=opt_c,
    optimizer_d=opt_d,
    train_set=train_set,
    lr_c=1.0,
    lr_d=1.0,
    train_acc_period=100,
    n_epoch=5,
    save_period=10000,
    batch_size=8,
    num_samples=1000,
    use_cuda=True,
    pixel=(0, 0, 0),
    dataset_with_labels=True,
):
    """
    Algorithm that trains both networks at the same time
    * inputs :
      - model_c : the completion network
      - model_d : the discrimination network
      - alpha : parameter that weights the two losses for the completion network (pixel-wise loss and dicrimination network-based loss)
      - optimizer_c : the optimizer for the completion network
      - optimizer_d : the optimizer for the discrimination network
      - train_set : training dataset
      - lr_c : learning rate for the completion network
      - lr_d : learning rate for the discrimination network
      - train_acc_period : intervals at which a loss is printed
      - n_epochs : number of epochs
      - save_period : intervals at which the models are automatically saved
      - batch_size : training batch size
      - num_samples : number of samples in the DataLoader sampler
      - use_cuda : boolean indicating the use of a GPU
      - pixel : the mean pixel on the training dataset
      - dataset_with_labels : boolean, True is the dataset used has labels
    """
    print("Beginning the adversarial training of the Completion and Discrimination networks")
    running_loss_c = 0
    running_loss_d = 0
    model_c.train()
    model_d.train()
    i = 0
    criterion = nn.BCELoss()
    for epoch in tqdm(range(n_epoch)):
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=6,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=num_samples),
            pin_memory=True,
            drop_last=True,
        )
        for data in tqdm(
            train_loader
        ):  # There is no need for labels in the training dataset, but we let users choose if they want to use labelized datasets
            if dataset_with_labels:
                x, _ = data
            else:
                x = data

            if use_cuda:
                x = x.cuda()

            # zero the parameter gradients

            opt_c.zero_grad()
            opt_d.zero_grad()

            # Training the completion network

            mask_c, centers_c = mask(batch_size, use_cuda=use_cuda)
            input_c = apply_mask(x, mask_c, pixel, use_cuda=use_cuda)
            output_c = model_c(input_c)
            loss_c_1 = C_loss(x, output_c, mask_c)

            # Feeding the discrimination network with completed images

            false = torch.zeros(
                (batch_size, 1)
            )  # The context discriminator should label all images as fake
            if use_cuda:
                false = false.cuda()
            input_gd_fake = (
                output_c.detach()
            )  # We detach here so that the gradients are not used, only doing a backward pass for D
            input_ld_fake = hole_cropping(input_gd_fake, centers_c, use_cuda=use_cuda)
            output_fake = model_d((input_ld_fake, input_gd_fake))
            loss_fake = criterion(output_fake, false)  # Loss if labelled as true

            # Feeding the discrimination network with real images

            centers_d = mask(batch_size, use_cuda=use_cuda, generate_mask=False)
            input_gd_real = x
            input_ld_real = hole_cropping(input_gd_real, centers_d, use_cuda=use_cuda)
            output_real = model_d((input_ld_real, input_gd_real))
            real = torch.ones((batch_size, 1))  # Tensor of ones, labels of all original images
            if use_cuda:
                real = real.cuda()
            loss_real = criterion(output_real, real)  # Loss if labelled as false

            # Updating the Discriminator model

            loss_d = alpha * (
                loss_real + loss_fake
            )  # Using a different coefficient for each kind of losses could be a way to personalize the net
            loss_d.backward()
            opt_d.step()

            # Computing the second loss for the completion network

            input_gd_fake = output_c  # Not detaching now because we will use the gradients
            input_ld_fake = hole_cropping(input_gd_fake, centers_c, use_cuda=use_cuda)
            output_fake = model_d((input_ld_fake, input_gd_fake))
            loss_c_2 = criterion(
                output_fake, real
            )  # Indeed, the completion network must lure the discrimination network, hence this loss

            # Updating the Completion model

            loss_c = loss_c_1 + alpha * loss_c_2
            loss_c.backward()
            opt_c.step()

            # printing some statistics

            running_loss_c = 0.33 * loss_c.item() / batch_size + 0.66 * running_loss_c
            running_loss_d = 0.33 * loss_d.item() / batch_size + 0.66 * running_loss_d
            if i % train_acc_period == train_acc_period - 1:
                print(
                    "[%d, %5d] loss for the completion network : %f" % (epoch + 1, i + 1, running_loss_c)
                )
                print(
                    "[%d, %5d] loss for the discrimination network : %f"
                    % (epoch + 1, i + 1, running_loss_d)
                )
                running_loss_c, running_loss_d = 0, 0
            i += 1
            del x
        if (epoch % save_period == save_period - 1) or (epoch == n_epoch - 1):
            filename1 = "model_c_save/model_c_checkpoint_c_and_d_training_epoch{:d}.pth.tar".format(
                epoch + 1
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_c.state_dict(),
                    "optimizer": opt_c.state_dict(),
                },
                filename=filename1,
            )
            filename2 = "model_d_save/model_d_checkpoint_c_and_d_training_epoch{:d}.pth.tar".format(
                epoch + 1
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_d.state_dict(),
                    "optimizer": opt_d.state_dict(),
                },
                filename=filename2,
            )
    print("Finished Training C&D simultaneously")
