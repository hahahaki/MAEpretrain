from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data.dataset
import numpy as np
# Example code to read a TIFF image

# For demonstration, let's assume the TIFF image is named "example.tiff"
# In practice, you would replace 'example.tiff' with the path to your actual TIFF file

image_path = '/home/codee/scratch/dataset/small_set/tensor1.tiff'  # Replace with the actual path to your TIFF file

# Open the image
with Image.open(image_path) as img:
    # Display the image
    plt.imshow(img)
    #print(img.size)
    plt.axis('off')  # Turn off axis numbers
    plt.show()
    path_list = ["/home/codee/scratch/dataset/small_set/tensor1.tiff",
    "/home/codee/scratch/dataset/small_set/tensor2.tiff",
    "/home/codee/scratch/dataset/small_set/tensor3.tiff"]
    data_list = []
    pad = 4
    for x in path_list:
    # Load the image using OpenCV
        image = Image.open(x)
        numpy_image = np.array(image)
    # Check if the image was successfully loaded

        tensor = torch.tensor(numpy_image)

        # Apply padding
        padded_tensor = F.pad(tensor, [pad for _ in range(4)])

        # Add the padded tensor to your data list
        data_list.append(padded_tensor)
    #print(key_name_dict)
    print(data_list)
    print(torch.cuda.is_available())