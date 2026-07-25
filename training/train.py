import sys
from pathlib import Path
from torch.optim import AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
import matplotlib.pyplot as plt
import lpips
import json


from DiT_model import DiT
from data import img_dataset
from VAE import VAE
from DDPM import LinearNoiseScheduler

from .train_script import train

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # older modules causes warnings, can safetly ignore

ROOT = Path(__file__).resolve().parent.parent


with open(f"{ROOT}/config/config.json", "r") as file:
        config = json.load(file)

device = "cuda" if torch.cuda.is_available() else "cpu"

vae = VAE(device = device, freeze = True, scaling_factor = config["VAE"]["scaling_factor"], path = f"{ROOT}/{config['Pretrained']['VAE']}").to(device)

dit = DiT(d_model             = config["DiT"]["d_model"],
            g_channels        = config["DiT"]["g_channels"],
            grid_size         = config["DiT"]["grid_size"],
            patch_size        = config["DiT"]["patch_size"],
            timestep_emb_dim  = config["DiT"]["timestep_emb_dim"],
            number_emb_dim    = config["DiT"]["number_emb_dim"],
            num_layers        = config["DiT"]["num_layers"],
            num_heads         = config["DiT"]["num_heads"],
            CLIP_dim          = config["DiT"]["CLIP_dim"])

scheduler = LinearNoiseScheduler(num_timesteps  = config["Scheduler"]["num_timesteps"],
                                    beta_start = config["Scheduler"]["beta_start"],
                                    beta_end   = config["Scheduler"]["beta_end"])


MSE_loss_fn = torch.nn.MSELoss()
clip_model = open_clip.create_model(
    "ViT-L-14",
    pretrained = f"{ROOT}/{config['Pretrained']['CLIP']}"
)
lpips_loss_fn = lpips.LPIPS(net='vgg', model_path = f"{ROOT}/{config['Pretrained']['vgg']}", verbose = False)

optimizer = AdamW(dit.parameters(), lr=config["Training"]["learning_rate"], weight_decay=0)

scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['Training']['epochs'])

dataset = img_dataset(root_dir = f"{ROOT}/{config['Data']['dataset']}", img_size = 224)
dataloader = DataLoader(dataset, batch_size = config["Training"]["batch_size"], shuffle = True)

print("modules loaded")

train(vae, dit, scheduler, MSE_loss_fn, clip_model, lpips_loss_fn, optimizer, scheduler_lr, dataloader)