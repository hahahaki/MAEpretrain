import torch
from PIL import Image
from torchvision.transforms import ToTensor
import os
import torchvision.transforms.functional as F


def pad_right_bottom(img, target_width, target_height):
    width, height = img.size
    padding_right = target_width - width
    padding_bottom = target_height - height
    return F.pad(img, (0, 0, padding_right, padding_bottom), fill=0)


def load_image(image_path):
    with Image.open(image_path) as img:
        return ToTensor()(img)


if __name__ == "__main__":
    
    base_directory = "/home/codee/scratch/dataset/cem500k"
    print("python start")
    
# Forming the list with file paths
# change the number for samples
    image_paths = [
        f"{base_directory}/{filename}" for filename in os.listdir(base_directory)
    ]
    
    path_list = image_paths

# Define the target size
    target_width, target_height = 224, 224

# List to store original sizes
    original_sizes = []

# List to store padded images
    padded_images = []
    tim = 1
    
    for path in image_paths:
        print("tim:", tim)
        f = open("/home/codee/scratch/mycode/myfile.txt", "w")
        f.write(str(tim) + ' ')
        tim = tim + 1
    # Open the image
        img = Image.open(path)
    # Store original size
        original_sizes.append(img.size)
    # Pad the image on the right and bottom
        img_padded = pad_right_bottom(img, target_width, target_height)
    # Convert to tensor and add to list
        padded_images.append(F.to_tensor(img_padded))

# Stack images for processing
    stacked_images = torch.stack(padded_images)

# Save the stacked tensor and original sizes
    torch.save(
        {"stacked_images": stacked_images, "original_sizes": original_sizes},
        "/home/codee/scratch/dataset/padded_images_and_sizes_500k.pt",
    )

    print("done")
