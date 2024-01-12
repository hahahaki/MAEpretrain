import random
import torch.utils.data.dataset
import torch.nn.functional as F
from monai.transforms import Compose, RandFlip, RandSpatialCrop, Resize
from utils import register_plugin, get_plugin
# I add
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from torchvision import transforms as tf
import os
from torch.utils.data import Dataset

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        radius = int(3*sigma)
        return tf.GaussianBlur(kernel_size=2 * radius + 1, sigma=sigma)(x)

class GaussNoise:
    """Gaussian Noise to be applied to images that have been scaled to fit in the range 0-1"""
    def __init__(self, var_limit=(1e-5, 1e-4), p=0.5):
        self.var_limit = np.log(var_limit)
        self.p = p
    
    def __call__(self, image):
        if np.random.random() < self.p:
            sigma = np.exp(np.random.uniform(*self.var_limit)) ** 0.5
            noise = np.random.normal(0, sigma, size=image.shape).astype(np.float32)
            image = image + torch.from_numpy(noise)
            image = torch.clamp(image, 0, 1)
        
        return image

'''
class EMData(Dataset):
    def __init__(
        self,
        data_dir,
        transforms,
        weight_gamma=None
    ):
        super(EMData, self).__init__()
        self.data_dir = data_dir
        
        self.subdirs = []
        #for sd in os.listdir(data_dir):
        #    if os.path.isdir(os.path.join(data_dir, sd)):
        #        self.subdirs.append(sd)
        
        # images and masks as dicts ordered by subdirectory
        self.paths_dict = [f"{data_dir}/{filename}" for filename in os.listdir(data_dir)]
        
        for sd in self.subdirs:
            sd_fps = glob(os.path.join(data_dir, f'{sd}/*.tiff'))
            if len(sd_fps) > 0:
                self.paths_dict[sd] = sd_fps
        

        # calculate weights per example, if weight gamma is not None
        self.weight_gamma = weight_gamma
        if weight_gamma is not None:
            self.weights = self._example_weights(self.paths_dict, gamma=weight_gamma)
        else:
            self.weights = None
        
        # unpack dicts to lists of images
         
        #for paths in self.paths_dict.values():
        #    self.paths.extend(paths)
        self.paths = self.paths_dict 
        print(f'Found {len(self.subdirs)} subdirectories with {len(self.paths)} images.')
        
        self.tfs = transforms
        
    def __len__(self):
        return len(self.paths)
    
    @staticmethod
    def _example_weights(paths_dict, gamma=0.3):
        # counts by source subdirectory
        counts = np.array(
            [len(paths) for paths in paths_dict.values()]
        )
        
        # invert and gamma the distribution
        weights = 1 / counts
        weights = weights ** gamma
        
        # for interpretation, normalize weights 
        # s.t. they sum to 1
        total_weights = weights.sum()
        weights /= total_weights
        
        # repeat weights per n images
        example_weights = []
        for w,c in zip(weights, counts):
            example_weights.extend([w] * c)
            
        return torch.tensor(example_weights)
    
    def __getitem__(self, idx):
        #get the filepath to load
        f = self.paths[idx]#.compute()
        
        #load the image and add an empty channel dim
        image = Image.open(f)
            
        #transform the images
        image1 = self.tfs(image)
        #image2 = self.tfs(image)
        #return the two images as 1 tensor concatenated on
        #the channel dimension, we'll split it later
        return image1
'''

@register_plugin('transform', 'betaaug2D')
def betaaug(cfg):
    """
    A data augmentation function that crops, resizes and flips a 2D image randomly.

    Args:
        cfg (dict): A dictionary containing configuration parameters for the data augmentation.

    Returns:
        A Compose object that applies the following transformations to a 2D volume:
        - Randomly crops a region of interest (ROI) from the volume.
        - Resizes the cropped ROI to a specified volume size using bilinear interpolation.
        - Randomly flips the volume along any axis with a probability of 0.5.
    """
    # Extract the minimum size and volume size from the configuration dictionary.
    min_size = int(cfg["vol_size"] * 0.5)
    volume_size = cfg["vol_size"]
    
    # Define a Compose object that applies the following transformations to a 3D volume.
    compose = Compose(
        [RandSpatialCrop(roi_size=(min_size, min_size,)),
         Resize((volume_size, volume_size,), mode="bilinear",
                align_corners=False),
         RandFlip(prob=.5), ])
    
    normalize = tf.Normalize(mean = 0.57287007, std = 0.12740536)
    '''
    augmentation = tf.Compose([
        tf.Grayscale(3),
        tf.RandomApply([tf.RandomRotation(180)], p=0.5),
        tf.RandomResizedCrop(224, scale=(0.2, 1.)), # here to change the dimension of the final input image
        tf.ColorJitter(0.4, 0.4, 0.4, 0.1),
        tf.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        tf.Grayscale(1),
        tf.RandomHorizontalFlip(),
        tf.RandomVerticalFlip(),
        #RandSpatialCrop(roi_size=(min_size, min_size,)),
        #Resize((volume_size, volume_size,), mode="bilinear",
                #align_corners=False),
        #RandFlip(prob=.5), 
        tf.ToTensor(),
        GaussNoise(p=0.5),
        normalize
    ])
    '''
    
    # Return the Compose object.
    return compose


# Cheng does not need this when pretrain on cem500k, does on train.py
# needed to be replaced by cem500K
@register_plugin("dataset", "Cem500K")
class Cem500K(torch.utils.data.dataset.Dataset):
    """
    Args:
        cfg (dict): A dictionary containing configuration parameters for the dataset.

    Attributes:
        path_list (list): A list of paths to the input data files.
        pad (int): The amount of padding to add to the input data.
        aug (str): A string indicates data augmentation to the input data.
        vol_size (int): The size of the 2D slices to sample from the 3D volumes.
        key_name_dict (dict): A dictionary mapping file paths to the names of the tensors in the data files.
        data_list (list): A list of tensors representing the input data.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a randomly sampled 2D slice from the input data at the given index.

    """

    def __init__(self, cfg):

        base_directory = '/home/codee/scratch/dataset/cem500k'

        file_paths = [f"{base_directory}/{filename}" for filename in os.listdir(base_directory)]

        self.path_list = file_paths
        print("filenum:", len(self.path_list))
        #print(cfg["path_list"])
        # need to be modified if we would like readin all the images in the directory
        self.pad = cfg["pad"] 
        self.aug = get_plugin("transform", cfg["aug"])(cfg)
        self.vol_size = cfg["vol_size"]
        
        '''
        # make changes of read in method:
        for x in self.path_list:
            # Load the image using PIL
            image = Image.open(x)
            #print("readimage size:", image.size)
            # Check if the image was successfully loaded
            if image is not None:
                # Convert the image to a PyTorch tensor
                transform = tf.ToTensor()
                tensor_image = transform(image)
                self.data_list.append(tensor_image.squeeze(0))
            else:
                print(f"Error loading image from path: {x}")
        
        
        # Later, to recover the original size
        data = torch.load('/home/codee/scratch/dataset/padded_images_and_sizes.pt')
        stacked_images = data['stacked_images']
        original_sizes = data['original_sizes']

        for i, tensor in enumerate(stacked_images):
            original_width, original_height = original_sizes[i]           
            tensor_cropped = tensor[:, :original_height, :original_width]
            self.data_list.append(tensor_cropped.squeeze(0))
        '''

        
    def __len__(self):
        """
        !!!! here is so important
        Returns the number of samples in each epoch.
        """
        #return 100000
        return len(self.path_list)

    def __getitem__(self, idx):
        """
        must have this function to define the obtained data
        Returns a randomly sampled 2D slice from the input data at the given index.
        Returns:
            A tensor representing a randomly sampled 2D slice from the input data.
        """
        img_path = self.path_list[idx]

        # Load the image
        image = Image.open(img_path)
        if image is not None:
            # Convert the image to a PyTorch tensor
            transform = tf.ToTensor()
            tensor_image = transform(image)
        else:
            print(f"Error loading image from path")
            
        _, d_x, d_y = tensor_image.shape
        # cheng add:
        # [224, 80]->[80, 80], 
        # then bilinear resize to [224, 224]
        sidelen = min(d_x, d_y)
        if sidelen is d_x:
            x_sample = 0
            # randint: [left, right)
            y_sample = torch.randint(
                low=0, high=int(d_y - sidelen + 1), size=(1,)
            )
        if sidelen is d_y:
            y_sample = 0
            x_sample = torch.randint(
                low=0, high=int(d_x - sidelen + 1), size=(1,)
            )

        sample = tensor_image[0][
            x_sample : x_sample + sidelen - 1,
            y_sample : y_sample + sidelen - 1,
        ].unsqueeze(0)
        #to_pil_image = tf.ToPILImage()
        #sample = to_pil_image(sample)
        # smaple now [x, y]
        #return sample
        #sample.savefig('/home/codee/scratch/result/check12.20.png')
        return self.aug(sample)


    def sample_cord(self, data_idx):
        """
        Samples a 2D slice from the input data at the given index and axis.

        Args:
            data_idx (int): The index of the input data to sample from.

        Returns:
            A tensor representing a 2D slice sampled from the input data at the given index and axis.
        """
        data = self.data_list[data_idx] # get dataset
        data = data.unsqueeze(0)
        _, d_x, d_y = data.shape
        # cheng add:
        # [224, 80]->[80, 80], 
        # then bilinear resize to [224, 224]
        sidelen = min(d_x, d_y)
        if sidelen is d_x:
            x_sample = 0
            # randint: [left, right)
            y_sample = torch.randint(
                low=0, high=int(d_y - sidelen + 1), size=(1,)
            )
        if sidelen is d_y:
            y_sample = 0
            x_sample = torch.randint(
                low=0, high=int(d_x - sidelen + 1), size=(1,)
            )

        sample = data[0][
            x_sample : x_sample + sidelen - 1,
            y_sample : y_sample + sidelen - 1,
        ].unsqueeze(0)
        #to_pil_image = tf.ToPILImage()
        #sample = to_pil_image(sample)
        # smaple now [x, y]
        #return sample
        #sample.savefig('/home/codee/scratch/result/check12.20.png')
        return self.aug(sample)