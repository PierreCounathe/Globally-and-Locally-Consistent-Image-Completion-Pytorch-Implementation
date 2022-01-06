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

from models import *

if torch.cuda.is_available():
  device = torch.device('cuda')
  use_cuda = True
  print('running on GPU')
else:
  device = torch.device('cpu')
  use_cuda = False
  print('running on CPU')

lr_d = 1
lr_c = 1
if use_cuda:
    model_c = Completion().cuda()
    model_d = Discriminator().cuda()
else:
    model_c = Completion()
    model_d = Discriminator()
opt_d = torch.optim.Adadelta(model_d.parameters(), lr = lr_d) 
opt_c = torch.optim.Adadelta(model_c.parameters(), lr = lr_c)

transform = transforms.Compose([transforms.RandomResizedCrop((256,256), scale = (0.6666666666667,1.0) ,ratio=(1.0,1.0)), transforms.ToTensor()])
train_set = torchvision.datasets.ImageFolder(root='data/celeba_train', transform= transform)
test_set = torchvision.datasets.ImageFolder(root="./data/celeba_test", transform=transform)
dataset_with_labels = True
test_loader = torch.utils.data.DataLoader(test_set)

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# UTILS FUNCTIONS
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
def size_crop() :
    """
    Defines the transformation that is applied to the images in pre-processing.
    
    """
    return transforms.RandomResizedCrop((256,256),scale = (0.6666666666667,1.0) ,ratio=(1.0,1.0))

def imshow(img):
    """
    * inputs:
          - img : [3, H, W] tensor
    """
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def mask(b_size,h = 256,w = 256, h_range_mask = (96,128), w_range_mask = (96,128),
         num_holes = 1,use_cuda = True, generate_mask = True) :
    """
    Generates as many masks as the batch size. 
    If generate_mask is True, returns the masks: tensors of the same shape as the images, with a randomly chosen rectangle area of ones. Also returns the masks centers.
    Else, it does not generate the masks and only returns the masks centers, useful to crop the image and create a local image, input of the discriminator network. 
    This is useful when training the Discriminator network on its own.
    * inputs:
            - b_size : the batch size of the tensor to add the mask
            - h,w : the height and wide of the images (default = (256,256))
            - h_range_mask : int tuple of size 2 representing the min and max dimensions for the height of the
            holes. The first element must be smaller than the second which is smaller than h
            - w_range_mask : int tuple of size 2 representing the min and max dimensions for the wide of the
            holes. The first element must be smaller than the second which is smaller than w
            - num_holes : the number of desired holes in the mask
            - generate_mask : when true, will ouput a mask. If not, will only return the centers
    * returns:
            - m : A torch tensor of shape [b_size,1,256,256], representing the mask (if generate_mask is True)
            - centers : A list of int tuples of the estimated centers' coordinates of the holes (we'll need them for the localD crop)
    """
    if generate_mask :
      if use_cuda :
        m = torch.zeros((b_size,1,h,w)).cuda() # Initializing the mask
      else :
        m = torch.zeros((b_size,1,h,w))
    centers = [[] for j in range(b_size)]
    for i_holes in range(num_holes) : # Number of masks
      for i_batch in range(b_size) : # Generation of 1 mask by image
        h_mask,w_mask = r.randint(*h_range_mask),r.randint(*w_range_mask) # Acquiring hole size
        i,j = r.randint(0,h-h_mask),r.randint(0,w-w_mask) # Acquiring hole position
        centers[i_batch].append((i+h_mask//2,j+w_mask//2))
        if generate_mask :
          for x in range(h_mask) :
            for y in range(w_mask) :
              m[i_batch,0,i+x,j+y] = 1.0 # Filling the hole location with ones
    if generate_mask :
      return (m,centers)
    else :
      return(centers)

def apply_mask(x, m, pixel, use_cuda = True) :
  """
  From an image a mask and a pixel value, outputs an image with a hole corresponding to the mask, the color of the input pixel value.
  * inputs :
      - x : a [b_size,3,H,W] shape tensor
      - m : Tensor of shape [b_size,1,H,W], representing the mask
      - pixel : The pixel value to put in the masked area int (tuple or list) in range [0,255]
  
  * outputs :
      A [b_size,4,H,W] tensor with all images patched + the mask in the the 4 elements of dim = 1
  """
  b_size,_,h,w = x.shape
  if use_cuda :
    pix = (torch.tensor(pixel).view((1, 3, 1, 1))/255.0).cuda()
  else :
    pix = torch.tensor(pixel).view((1, 3, 1, 1))/255.0
  x_patched = x - x*m + m*pix
  return torch.cat((x_patched,m),dim=1)

def visuel_mask(test_loader, n_batch = 1, use_cuda = True) :
  """
  A visual function that will show what is a batch of images with turquoise holes
  * inputs :
    - test_loader : test DataLoader
    - n_batch : number of batches to show
    - use_cuda : cuda availability bool
  """
  for i in range(n_batch) :
    # get some random training images
    dataiter = iter(test_loader)
    images, _ = dataiter.next()
    turquoise_rgb = (64,224,208)
    m,l = mask(64,h=32,w=32,h_range_mask=(5,10),w_range_mask=(5,10), num_holes=1) # Generate an appropriate turquoise mask
    if use_cuda :
      im = apply_mask(images.cuda(),m, turquoise_rgb) # Apply the mask
    else :
      im = apply_mask(images,m, turquoise_rgb)
    # show images with the patches
    imshow(torchvision.utils.make_grid(im[:,:3]))
    imshow(torchvision.utils.make_grid(im[:,3:]))
    return(im[:,:3],l)

def hole_cropping(x,centers,use_cuda = True) : #This function is not so generic because its particular to our problem, it returns an image half the size of the first
  """
  Takes a batch image, hole centers coordinates and returns images half the size of the original ones, centered around the holes centers.
  * inputs :
      - x : a [b_size,3,H,W] shape tensor
      - centers : a list of list of int tuples representing a (b_size,1) shape array, containing the center of the holes
  
  * outputs :
      A [b_size,3,H//2,W//2] a cropped version of the image, the crop being centered on the hole location
  """
  b_size,_,h,w = x.shape
  if use_cuda :
    t = torch.zeros((b_size,3,h//2,w//2)).cuda()
  else :
    t = torch.zeros((b_size,3,h//2,w//2))
  for i in range(b_size) :
    h1,w1 = centers[i][0] #Only take the first hole of the list, assuming there is only one hole.
    h1 = min(3*h//4,max(h//4,h1)) #If the center is too close from the edges, the holes won't be in the center
    w1 = min(3*w//4,max(w//4,w1))
    t[i] = x[i,:,h1-h//4:h1+h//4,w1-w//4:w1+w//4]
  return t

def test_and_compare(model = model_c, model_d = model_d, test_loader = test_loader, dataset_with_labels = True, number_of_pictures = 5, h_range_mask = (96,128), w_range_mask = (96,128),
         num_holes = 1, p = 0.002, pixel = (64,224,208), savefigures = False):
  """
  Shows images from the testloader. Original image, image with mask, and the completed one.
  The title contains the outputs of the discrimination network for the original image and for the completed one, for the task of classifying images as original (1) or completed (0).
  * inputs :
    - model_c : trained completion network
    - model_d : trained discrimination network
    - test_loader : a test DataLoader iterator
    - dataset_with_labels : boolean, True is the dataset used has labels
    - number_of_pictures : the number of pictures to plot
    - h_range_mask : the range of mask heights
    - w_range_mask : the range of mask widths
    - num_holes : number of holes per image
    - p : an arbitraty probability to artificially select random images from a DataLoader iterator
    - pixel : the mean pixel in the training dataset
    - savefigures : Boolean, if true, saves the output figures

  """
  model.eval()
  model_d.eval()
  i=0
  for data in test_loader:
    if np.random.random() < p :
      if i < number_of_pictures:
        if dataset_with_labels:
          X, Y= data
        else:
          X = data
        for x in X:
          m, centers =mask(b_size=1, h_range_mask = h_range_mask, w_range_mask = w_range_mask,
         num_holes = num_holes)
          centers_true =mask(b_size=1, h_range_mask = h_range_mask, w_range_mask = w_range_mask,
         num_holes = num_holes, generate_mask = False)
          plt.figure(figsize = [15, 5])
          # imshow((x.cuda().view(3, 256, 256)))
          image1 = (x.cuda().view(3, 256, 256))
          ax1 = plt.subplot(1, 3, 1)
          npimg1 = (image1.cpu().numpy())
          plt.imshow(np.transpose(npimg1, (1, 2, 0)))
          # imshow((x.cuda() - x.cuda()* m).view(3, 256, 256))
          image2 = (x.cuda() - x.cuda()* m).view(3, 256, 256)
          ax2 = plt.subplot(1, 3, 2)
          npimg2 = (image2.cpu().numpy())
          plt.imshow(np.transpose(npimg2, (1, 2, 0)))
          x = x.view(1, 3, 256, 256) # to apply the mask : x is equivalent to a batch of size 1
          if use_cuda:
            im = apply_mask(x.cuda(), m, pixel).view(4, 256, 256)
          else:
            im = apply_mask(x, m, pixel).view(4, 256, 256)
          model_c_input = im.view(1, 4, 256, 256)
          model_c_output = model(model_c_input)
          recombined_output = x.cuda() - x.cuda() * m + model_c_output * m
          # imshow(recombined_output.detach().view(3, 256, 256))
          image3 = recombined_output.detach().view(3, 256, 256)
          ax3 = plt.subplot(1, 3, 3)
          npimg3 = (image3.cpu().numpy())
          input_true_ld = hole_cropping(x.cuda().detach(),centers_true,use_cuda = use_cuda)
          output_true_d = model_d((input_true_ld,x.cuda().detach()))
          prob_t = output_true_d.item()
          input_ld = hole_cropping(model_c_output,centers,use_cuda = use_cuda)
          output_d = model_d((input_ld,model_c_output))
          prob = output_d.item()
          plt.title("Original Image discrimination network score (1 = classified as original) : {} - Completed Image discrimination network score (1 = classified as original) : {}".format(prob_t,prob), loc = 'center')
          plt.imshow(np.transpose(npimg3, (1, 2, 0)))
          if savefigures:
            plt.savefig('saves/figures/fig{}.png'.format(int(time.time()))) #Choose your directory
      else: 
        break
      i+=1
  
def save_checkpoint(state, filename):
  """
  Saves the state checkpoint of a model in a directory
  """
  print('=> Saving {}'.format(filename))
  torch.save(state, filename)

def load_checkpoint(model, optimizer, filename):
  """
  Loads the model and optimizer contained in a file, into the model and optimizer passed in the function.
  *inputs:
  - model : a model initialized with the corresponding class
  - optimizer : an adadelta optimizer initialized with the model parameters
  - filename : the filename of the model to be loaded
  """
  print('=> Loading {} into the model'.format(filename))
  checkpoint = torch.load(filename)
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  print("Model and optimizer loaded")

def mean_pixel(dataset = train_set, dataset_with_labels = True) :
  """
  Computes the mean pixel of the train dataset that serve as base masks color.
  * inputs : 
    - dataset : the train dataset
  $ ouputs : 
    - tuple(pix) : a tuple (R, G, B) of the values of the RGB channels of the mean pixel
  """
  pix = torch.zeros(3)
  n = len(dataset)
  for i in tqdm(range(n)) :
    data = dataset[i]
    if dataset_with_labels :
      data = data[0]
    r = data[0].mean()
    g = data[1].mean()
    b = data[2].mean()
    t = torch.tensor([r,g,b])
    pix += t
  pix = (255.0/n) * pix
  return(tuple(pix))
