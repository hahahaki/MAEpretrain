import random
import torch.utils.data.dataset
import torch.nn.functional as F
from monai.transforms import Compose, RandFlip, RandSpatialCrop, Resize
from utils import register_plugin, get_plugin
# I add
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
import os


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
    
    # Return the Compose object.
    return compose

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

        # Forming the list with file paths
        # change the number for samples
        file_paths = [f"{base_directory}/{filename}" for filename in os.listdir(base_directory)]
        self.path_list = file_paths
        #print(cfg["path_list"])
        # need to be modified if we would like readin all the images in the directory
        self.pad = cfg["pad"] 
        self.aug = get_plugin("transform", cfg["aug"])(cfg)
        self.vol_size = (
            cfg["vol_size"] + int(cfg["vol_size"] // cfg["patch_size"])
            if cfg["patch_size"] % 2 == 0
            else cfg["vol_size"]
        ) 
        self.data_list = []
        '''
        # ex. The key "data/folder1/tensor1_tensor.pkl" maps to "tensor1".
        self.key_name_dict = {
            x: x.split("/")[-1].split("_tensor")[0] for x in self.path_list
        }
        self.data_list = [
            F.pad(torch.load(x)[self.key_name_dict[x]], [self.pad for _ in range(6)])
            for x in self.path_list
        ]
        '''
        
        # make changes of read in method:
        for x in self.path_list:
            # Load the image using PIL
            image = Image.open(x)
            #print("readin size:", image.size)
            '''
            # Check if the image dimensions are 224x224
            if image.size != (224, 224):
                # Calculate padding
                delta_width = 224 - image.size[0]
                delta_height = 224 - image.size[1]
                padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        
                # Apply padding
                image = ImageOps.expand(image, padding)  
            '''
            # Check if the image was successfully loaded
            if image is not None:
                # Convert the image to a PyTorch tensor
                #numpy_image = np.array(image)
                #tensor = torch.tensor(numpy_image)
                # Apply padding, range(4): up, down, left, right
                #padded_tensor = F.pad(tensor, [self.pad for _ in range(4)])
                # Add the padded tensor to your data list
                # the dimension is [1, H, W]
                #print(padded_tensor)
                transform = transforms.ToTensor()
                tensor_image = transform(image)
                self.data_list.append(tensor_image.squeeze(0))
            else:
                print(f"Error loading image from path: {x}")

    def __len__(self):
        """
        !!!! here is so important
        Returns the number of samples in each epoch.
        """
        return 500000

    def __getitem__(self, idx):
        """
        must have this function to define the obtained data
        Returns a randomly sampled 2D slice from the input data at the given index.
        Returns:
            A tensor representing a randomly sampled 2D slice from the input data.
        """
        curr_data_idx = random.randrange(0, len(self.data_list)) # select dataset
        # return self.data_list[curr_data_idx].unsqueeze(0)
        return self.sample_cord(curr_data_idx) # return 2D slice


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

        x_sample = torch.randint(
            low=0, high=int(d_x - self.vol_size - 1), size=(1,)
        )
        y_sample = torch.randint(
            low=0, high=int(d_y - self.vol_size - 1), size=(1,)
        )
        sample = data[0][
            x_sample : x_sample + self.vol_size,
            y_sample : y_sample + self.vol_size,
        ].unsqueeze(0)
        # smaple now [x, y]
        #return sample
        return self.aug(sample)