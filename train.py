import os
import torch.distributed as dist
import torch.utils.data.distributed
from modell import *
import numpy
import random
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pretrain_engine import set_deterministic
from utils import get_plugin, read_yaml, save_checkpoint
from dataset import * # I added this, so @register in dataset.py can be added


parser = argparse.ArgumentParser(description="MAESTER Training")
parser.add_argument("--model_config_dir", default="/home/codee/scratch/mycode", type=str)
parser.add_argument("--model_config_name", default="default.yaml", type=str)
parser.add_argument("--dist_backend", default="nccl", type=str, help="")
# for debug, keep it small, the number of GPUs
parser.add_argument("--world_size", default=1, type=int, help="")
# IP address in a distributed setting implies that the communication is intended to happen internally within the same machine
parser.add_argument("--init_method", default="tcp://127.0.0.1:56079", type=str, help="")
parser.add_argument("--logdir", default="/home/codee/scratch/checkpoints", type=str, help="log directory")

print("Starting...", flush=True)
args = parser.parse_args()
set_deterministic()
#print(args.model_config_dir)
#print(args.model_config_name)
cfg = read_yaml(os.path.join(args.model_config_dir, args.model_config_name))
#print("model_config:", cfg, flush=True)

ngpus_per_node = torch.cuda.device_count()


print("availalbecuda:", torch.cuda.is_available())


# this line is for the GPU in cluster 
current_device = local_rank = int(os.environ.get("SLURM_LOCALID"))
rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
torch.cuda.set_device(rank)

# this is for single GPU
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
else:
    current_device = torch.device("cpu")

print("current:", current_device)

# initialize the DistributedSampler
# this is for distribution training

dist.init_process_group(
    backend=args.dist_backend,
    init_method=args.init_method,
    world_size=args.world_size,
    rank=rank,
)


print(f"From Rank: {rank}, ==> Process group ready!", flush=True)
print(f"From Rank: {rank}, ==> Building model..")
model = get_plugin("model", cfg["MODEL"]["name"])(cfg["MODEL"])
#this is for the GPU is possible
if torch.cuda.is_available():
    model = model.cuda()

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])
model.embed_dim = cfg["MODEL"]["embed_dim"]
print(f"From Rank: {rank}, ==> Model ready!")
print(f"From Rank: {rank}, ==> Preparing data..")


'''
normalize = tf.Normalize(mean = 0.57287007, std = 0.12740536)
augmentation = tf.Compose([
    tf.Grayscale(3),
    tf.RandomApply([tf.RandomRotation(180)], p=0.5),
    tf.RandomResizedCrop(96, scale=(0.2, 1.)),
    tf.ColorJitter(0.4, 0.4, 0.4, 0.1),
    tf.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    tf.Grayscale(1),
    tf.RandomHorizontalFlip(),
    tf.RandomVerticalFlip(),
    tf.ToTensor(),
    GaussNoise(p=0.5),
    normalize
])
'''
dataset = get_plugin("dataset", cfg["DATASET"]["name"])(cfg["DATASET"])
#path = '/home/codee/scratch/dataset/small_set'
#dataset = EMData(path, augmentation)
print("Total images in dataset:", len(dataset))
'''
for i, sample in enumerate(dataset):
    # Assuming sample is a tuple of (image, label)
    image, _ = sample

    # Print the shape of the image tensor
    print(f"Sample {i}: Shape - {image.shape}")

    # Check only the first few samples
    if i >= 5:  # Check first 5 samples
        break
'''
# determinstic behaviour
def seed_worker(worker_id):
    set_deterministic()


g = torch.Generator()
g.manual_seed(0)

# this is for ditribution training
sampler = DistributedSampler(dataset)
print("size:", cfg["DATASET"]["batch_size"])
dataloader = DataLoader(
    dataset,
    # the batch can be automatically ajusted
    batch_size=cfg["DATASET"]["batch_size"],
    #sampler is for distribution train
    sampler=sampler,
    num_workers=cfg["DATASET"]["num_workers"],
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)

optimizer = get_plugin("optim", cfg["OPTIM"]["name"])(model, cfg["OPTIM"])

print(f"From Rank: {rank}, ==> Data ready!")
engine_func = get_plugin("engine", cfg["ENGINE"]["name"])

for epoch in range(cfg["ENGINE"]["epoch"]):
    epoch_loss = engine_func(model, dataloader, optimizer, cfg)
    print("loss:", epoch_loss) # the average loss across each batch
    print("step:", epoch)
    if (epoch + 1) % 20 == 0:
        # this is for GPUs
        state_dict = model.module.state_dict()
        #state_dict = model.state_dict()
        save_checkpoint(args.logdir, state_dict, name="pretrain500k1_8.pt")
    
    f = open("/home/codee/scratch/mycode/myfile.txt", "w")
    f.write(str(epoch) + ' ')
print(f"From Rank: {rank}, ==> Training finished!")
