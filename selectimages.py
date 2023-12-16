from PIL import Image
import os
import numpy as np

def process_images(source_dir, destination_dir, max_files=200):
    """
    Read the first 100 image files from the source directory, and
    write them to the destination directory with new names.

    Args:
    source_dir (str): The path of the source directory containing image files.
    destination_dir (str): The path of the destination directory to save new images.
    max_files (int): Maximum number of files to process.
    """
    os.makedirs(destination_dir, exist_ok=True)
    file_counter = 0

    # Iterate over files in the source directory
    for filename in os.listdir(source_dir):
        if file_counter >= max_files:
            break
        #minn = 10000
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            src_file_path = os.path.join(source_dir, filename)
            try:
                # Open the image file
                img = Image.open(src_file_path)
                np_image = np.array(img)
                if np_image.shape[0] < 100 or np_image.shape[0] < 100:
                    continue
                print(np_image.shape)
                #print(f"Image: {path}")
                print(f"Min pixel value: {np_image.min()}")
                print(f"Max pixel value: {np_image.max()}")
                print(f"Mean pixel value: {np_image.mean()}")
                print(f"Standard deviation: {np_image.std()}")
                #print(src_file_path)
                # Save the image with a new name in the destination directory
                new_filename = f"tensor{file_counter}.tiff"
                dst_file_path = os.path.join(destination_dir, new_filename)
                #print(dst_file_path)
                img.save(dst_file_path)

                file_counter += 1
            except IOError:
                print(f"Cannot read file {filename}")

# Source directory containing image files
source_directory = "/home/codee/scratch/dataset/cem500k"

# Destination directory to save new images
destination_directory = "/home/codee/scratch/dataset/small_set"

# Process the images
process_images(source_directory, destination_directory)

print("finish")