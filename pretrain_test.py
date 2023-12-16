import torch
from modell import *
from utils import get_plugin, read_yaml, save_checkpoint
import os 
import argparse
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

#print(args.model_config_dir)
#print(args.model_config_name)
parser = argparse.ArgumentParser(description="MAESTER Training")
parser.add_argument("--model_config_dir", default="/home/codee/scratch/mycode", type=str)
parser.add_argument("--model_config_name", default="default.yaml", type=str)
parser.add_argument("--dist_backend", default="nccl", type=str, help="")
parser.add_argument("--world_size", default=2, type=int, help="")
parser.add_argument("--init_method", default="tcp://127.0.0.1:56079", type=str, help="")
parser.add_argument("--logdir", default="/home/codee/scratch/checkpoints", type=str, help="log directory")

print("Starting...", flush=True)
args = parser.parse_args()
cfg = read_yaml(os.path.join(args.model_config_dir, args.model_config_name))

model = get_plugin("model", cfg["MODEL"]["name"])(cfg["MODEL"])

# Load the trained state dictionary
model.load_state_dict(torch.load('/home/codee/scratch/checkpoints/latest4.pt'))

# Set the model to evaluation mode
#print(model.eval())

# Load and transform the image
img_path = '/home/codee/scratch/dataset/small_set/tensor50.tiff'
image = Image.open(img_path)

#print(image.size)
pad = 8
numpy_image = np.array(image)
tensor = torch.tensor(numpy_image)
#print("tensor:", tensor.shape)
# Apply padding, range(4): up, down, left, right
padded_tensor = F.pad(tensor, [pad for _ in range(4)])
# should be [batch_size, C, H, W] dimension
input_tensor = padded_tensor.unsqueeze(0).unsqueeze(0)
#print("squeeze:", input_tensor.shape)
# Add the padded tensor to your data list
# Make a prediction
with torch.no_grad():
    loss, pred, mask = model(input_tensor, mask_ratio=cfg["MODEL"]["mask_ratio"])

# patchify
p = 8
h = w = 30
x = input_tensor.reshape(shape=(input_tensor.shape[0], 1, h, p, w, p))   
x = torch.einsum("nchpwq->nhwpqc", x)
x = x.reshape(shape=(input_tensor.shape[0], h * w, p**2 * 1))
# unpachify

x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
x = torch.einsum("nhwpqc->nchpwq", x)
imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))




# Process the prediction (specific to your model's output)
# For example, if it's a classification model, apply softmax
# predicted_class = torch.argmax(torch.nn.functional.softmax(prediction, dim=1)).item()
print("loss:", loss)
#print(pred.shape)
#print(mask.shape)
# Print or use the prediction
def save_tensor_as_image(image, file_name):
    plt.imshow(image, cmap='gray')
    #print(image)
    plt.axis('off')  # Turn off axis numbers and labels
    plt.savefig(file_name)
    plt.close()

# unpachify, change the parameters
p = 8
h = w = int(900 ** 0.5)
pred = pred.reshape(shape=(1, h, w, p, p, 1))
pred = torch.einsum("nhwpqc->nchpwq", pred)
img = pred.reshape(shape=(pred.shape[0], 1, h * p, h * p))
print(img[0][0])
save_tensor_as_image(padded_tensor, '/home/codee/scratch/result/input.png')
save_tensor_as_image(img[0][0], '/home/codee/scratch/result/pred.png')
save_tensor_as_image(mask, '/home/codee/scratch/result/mask.png')

save_tensor_as_image(imgs[0][0], '/home/codee/scratch/result/test.png')