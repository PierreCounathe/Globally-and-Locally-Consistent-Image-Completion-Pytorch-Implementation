import torch
import torchvision
from torchvision.transforms import RandomResizedCrop
import numpy as np
import matplotlib.pyplot as plt
import random as r
from tqdm import tqdm
import os

if os.path.exists("saves/figures"):
    NUM_FIGURES = len(os.listdir("saves/figures"))


def size_crop():
    """
    Defines the transformation that is applied to the images in pre-processing.

    Returns:
        RandomResizedCrop: A pre-processing transformation.
    """

    return RandomResizedCrop((256, 256), scale=(0.6666666666667, 1.0), ratio=(1.0, 1.0))


def imshow(img: torch.Tensor):
    """Simple visualisation function for torch tensors.

    Args:
        img (torch.Tensor): The tensor representation of the image.
    """
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def mask(
    b_size,
    h=256,
    w=256,
    h_range_mask=(96, 128),
    w_range_mask=(96, 128),
    num_holes=1,
    device=torch.device("cpu"),
    generate_mask=True,
):
    """
    Generates a mask tensor (of values 1 and 0) and a list of each holes centers.

    Args:
            - b_size : the batch size of the tensor to add the mask
            - h,w : the height and wide of the images (default = (256,256))
            - h_range_mask : int tuple of size 2 representing the min and max dimensions for the height of the
            holes. The first element must be smaller than the second which is smaller than h
            - w_range_mask : int tuple of size 2 representing the min and max dimensions for the wide of the
            holes. The first element must be smaller than the second which is smaller than w
            - num_holes : the number of desired holes in the mask
            - generate_mask : when true, will ouput a mask. If not, will only return the centers
    Returns:
            - m : A torch tensor of shape [b_size,1,256,256], representing the mask (if generate_mask is True)
            - centers : A list of int tuples of the estimated centers' coordinates of the holes (we'll need them for the localD crop)
    """
    if generate_mask:

        # Initializing the mask.
        m = torch.zeros((b_size, 1, h, w)).to(device)

    # Initializing the center.
    centers = [[] for _ in range(b_size)]

    for _ in range(num_holes):
        for i_batch in range(b_size):

            # Acquiring hole size.
            h_mask, w_mask = r.randint(*h_range_mask), r.randint(*w_range_mask)

            # Acquiring hole position.
            i, j = r.randint(0, h - h_mask), r.randint(0, w - w_mask)

            # Writing hole coordinates.
            centers[i_batch].append((i + h_mask // 2, j + w_mask // 2))

            if generate_mask:

                # Filling the hole location with ones.
                m[i_batch, :, i : i + h_mask, j : j + w_mask] = 1.0

    return (m, centers) if generate_mask else centers


def apply_mask(x, m, pixel, device=torch.device("cpu")):
    """
    Set each pixel in the masked zone of an image to a given color.

    Args:
        - x : a [b_size,3,H,W] shape tensor
        - m : Tensor of shape [b_size,1,H,W], representing the mask
        - pixel : The pixel value to put in the masked area int (tuple or list) in range [0,255]

    Returns:
        A [b_size,4,H,W] tensor with all images patched + the mask in the the 4 elements of dim = 1
    """
    pix = torch.tensor(pixel).view((1, 3, 1, 1)) / 255.0
    pix = pix.to(device)

    x_patched = x - x * m + m * pix

    return torch.cat((x_patched, m), dim=1)


def visuel_mask(test_loader, n=1, device=torch.device("cpu")):
    """
    A visual function that will show what is a batch of images with turquoise holes
    """
    for _ in range(n):
        # Get some random training images.
        dataiter = iter(test_loader)
        images, _ = dataiter.next().to(device)

        # Generate an appropriate mask.
        m, l = mask(64, h=32, w=32, h_range_mask=(5, 10), w_range_mask=(5, 10), num_holes=1, device=device)
        im = apply_mask(images, m, (64, 224, 208))

        # Show images with the patches.
        imshow(torchvision.utils.make_grid(im[:, :3]))
        imshow(torchvision.utils.make_grid(im[:, 3:]))

        return (im[:, :3], l)


def hole_cropping(x, centers, device=torch.device("cpu")):
    """
    * inputs :
        - x : a [b_size,3,H,W] shape tensor
        - centers : a list of list of int tuples representing a (b_size,1) shape array, containing the center of the holes

    * outputs :
        A [b_size,3,H//2,W//2] a cropped version of the image, the crop being centered on the hole location
    """
    b_size, _, h, w = x.shape
    t = torch.zeros((b_size, 3, h // 2, w // 2)).to(device)

    for i in range(b_size):

        # Only take the first hole of the list, assuming there is only one hole.
        h1, w1 = centers[i][0]

        # If the center is too close from the edges, the holes won't be in the center.
        h1 = min(3 * h // 4, max(h // 4, h1))
        w1 = min(3 * w // 4, max(w // 4, w1))

        # Crop the image.
        t[i] = x[i, :, h1 - h // 4 : h1 + h // 4, w1 - w // 4 : w1 + w // 4]

    return t


def test_and_compare(
    model_c,
    model_d,
    test_loader,
    number_of_pictures=5,
    h_range_mask=(96, 128),
    w_range_mask=(96, 128),
    num_holes=1,
    p=0.002,
    dataset_with_labels=False,
    device=torch.device("cpu"),
    pixel=(130, 107, 95),
):
    """
    Shows images from the testloader. Original image, with mask, and the completed one.

    """
    model_c.eval()
    model_d.eval()
    i = 0
    global NUM_FIGURES
    for data in test_loader:
        if i < number_of_pictures:
            # Random number used to use random images from test set.
            if np.random.random() > p:
                continue
            if dataset_with_labels:
                X, _ = data
            else:
                X = data
            for x in X:
                x = x.to(device)
                m, centers = mask(
                    b_size=1,
                    h_range_mask=h_range_mask,
                    w_range_mask=w_range_mask,
                    num_holes=num_holes,
                    device=device,
                )
                centers_true = mask(
                    b_size=1,
                    h_range_mask=h_range_mask,
                    w_range_mask=w_range_mask,
                    num_holes=num_holes,
                    generate_mask=False,
                    device=device,
                )
                plt.figure(figsize=[15, 5])
                # imshow((x.view(3, 256, 256)))
                image1 = x.view(3, 256, 256)
                ax1 = plt.subplot(1, 3, 1)
                npimg1 = image1.cpu().numpy()
                plt.imshow(np.transpose(npimg1, (1, 2, 0)))
                # imshow((x - x* m).view(3, 256, 256))
                image2 = (x - x * m).view(3, 256, 256)
                ax2 = plt.subplot(1, 3, 2)
                npimg2 = image2.cpu().numpy()
                plt.imshow(np.transpose(npimg2, (1, 2, 0)))

                # to apply the mask : x is equivalent to a batch of size 1
                x = x.view(1, 3, 256, 256).to(device)
                im = apply_mask(x, m, pixel).view(4, 256, 256)
                model_c_input = im.view(1, 4, 256, 256)
                model_c_output = model_c(model_c_input)
                recombined_output = x.to(device) - x * m + model_c_output * m
                # imshow(recombined_output.detach().view(3, 256, 256))
                image3 = recombined_output.detach().view(3, 256, 256)
                ax3 = plt.subplot(1, 3, 3)
                npimg3 = image3.cpu().numpy()
                input_true_ld = hole_cropping(x.detach(), centers_true, device=device)
                output_true_d = model_d((input_true_ld, x.detach()))
                prob_t = output_true_d.item()
                input_ld = hole_cropping(model_c_output, centers, device=device)
                output_d = model_d((input_ld, model_c_output))
                prob = output_d.item()
                plt.title(f"P(True) = True : {prob_t:3f} - P(False) = True : {prob:3f}", loc="center")
                plt.imshow(np.transpose(npimg3, (1, 2, 0)))
                plt.savefig("saves/figures/fig{}.png".format(NUM_FIGURES + 1))
                NUM_FIGURES += 1
            i += 1
        else:
            break


def pixel_moyen(dataset, dataset_with_labels=False):
    """Compute the mean pixel RGB value of a dataset.

    Args:
        dataset (iterable containing the RGB img): The dataset to compute the mean pixel.
        dataset_with_labels (bool, optional): Whether the dataset has a label with images. Defaults to False.

    Returns:
        tuple: The RGB mean pixel.
    """
    pix = torch.zeros(3)
    n = len(dataset)
    for i in tqdm(range(n)):
        data = dataset[i]
        if dataset_with_labels:
            data = data[0]
        r = data[0].mean()
        g = data[1].mean()
        b = data[2].mean()
        t = torch.tensor([r, g, b])
        pix += t
    pix = (255.0 / n) * pix

    return tuple(pix)


def save_checkpoint(state, filename):
    """
    Save the state checkpoint in a directory
    """
    print("=> Saving {}".format(filename))
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, device=torch.device("cpu")):
    """
    Loads the model and optimizer contained in a file, into the model and optimizer passed in the function.
    *inputs:
    - model : a model initialized with the corresponding class
    - optimizer : an adadelta optimizer initialized with the model parameters
    - filename : the filename of the model to be loaded
    """
    print("=> Loading {} into the model".format(filename))
    device = torch.device("cuda:0") if (torch.cuda.is_available()) else torch.device("cpu")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Model and optimizer loaded")
