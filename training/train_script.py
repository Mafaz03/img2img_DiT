import sys
from pathlib import Path
from torch.optim import AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
import matplotlib.pyplot as plt
import lpips
import json

sys.path.append(str(Path().resolve().parent))
str(Path().resolve().parent)


from DiT_model import DiT
from data import img_dataset
from VAE import VAE
from DDPM import LinearNoiseScheduler

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # older modules causes warnings, can safetly ignore

ROOT = Path(__file__).resolve().parent.parent


def train(vae           : torch.nn.Module, 
          dit           : torch.nn.Module, 
          scheduler     : LinearNoiseScheduler, 
          MSE_loss_fn   : torch.nn.MSELoss,
          clip_model    : torch.nn.Module,
          lpips_loss_fn : torch.nn.Module,
          optimizer     : torch.optim,
          dataloader    : torch.utils.data.DataLoader
          ):

    with open(f"{ROOT}/config/config.json", "r") as file:
        config = json.load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    for param in vae.parameters():
        param.requires_grad = False

    for param in clip_model.parameters():
        param.requires_grad = False

    vae = vae.to(device)
    clip_model = clip_model.to(device)
    dit = dit.to(device) 
    lpips_loss_fn = lpips_loss_fn.to(device)

    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
    CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)

    def clip_preproces(img):
        # images are normalised [-1, 1] for VAE and DiT
        img = (img + 1)/2
        if (img.shape[-1] != 224) or (img.shape[-2] != 224):
            img = torch.nn.functional.interpolate(img, size=224, mode='bicubic', align_corners=False)
        img = (img - CLIP_MEAN) / CLIP_STD
        return img

    losses = []
    start_epoch = 0

    lambda_diffuion = config["weights"]["diffusion"]
    lambda_clip     = config["weights"]["CLIP"]
    lambda_lpips    = config["weights"]["lpips"]
    acc_steps       = config["Training"]["accumulation_step"]

    epochs = config["Training"]["epochs"]

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        step_count = 0
        for condition_image, actual_image in dataloader:
            step_count += 1

            condition_image = condition_image.to(device)
            actual_image    = actual_image.float().to(device)

            ######################
            # DIFFUSION MSE LOSS #
            ######################

            # actual image VAE-ing
            with torch.no_grad():
                mu, logvar = vae.encode(actual_image)
                z = vae.reparameterize(mu, logvar)   # [B, 4, 28, 28]

            # condition image VAE-ing
            with torch.no_grad():
                mu_c, logvar_c = vae.encode(condition_image)
                z_c = vae.reparameterize(mu_c, logvar_c) # [B, 4, 28, 28]

            # Sample random noise
            noise = torch.randn_like(z).to(device)   # [B, 4, 28, 28]

            B = z.shape[0]
            # Sample timestep
            t = torch.randint(0, 1000,(B,)).to(device) # [B]

            # adding nooise to actual image
            noisy_im = scheduler.add_noise(z, noise, t) # [B, 4, 28, 28]

            # CLIP encoding
            clip_condition = clip_model.encode_image(clip_preproces(condition_image)) # [B, 768]

            # DiT Prediction (in latent space)
            dit_pred = dit(noisy_im, t, z_c, clip_condition)

            diffusion_mse_loss = MSE_loss_fn(dit_pred, noise)


            #############
            # CLIP LOSS #
            #############

            # CLIP encoding for actual image
            with torch.no_grad():
                target_emb = clip_model.encode_image(clip_preproces(actual_image))

            t_clip = torch.randint(0, 200, (B,)).to(device)
            # adding nooise to actual image
            noisy_im = scheduler.add_noise(z, noise, t_clip) # [B, 4, 28, 28]

            # DiT Prediction (in latent space)
            dit_pred = dit(noisy_im, t_clip, z_c, clip_condition)

            x0_pred = scheduler.get_x0(noisy_im, dit_pred, t_clip)

            # decoding image to pixel space
            predicted_image = vae.decode(x0_pred)

            # encoding the predicted image
            pred_emb = clip_model.encode_image(clip_preproces(predicted_image))

            pred_emb   = pred_emb / pred_emb.norm(dim=-1, keepdim=True)
            target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True)

            clip_loss = 1 - (pred_emb * target_emb).sum(dim=-1).mean()   # 1 - cosine similarity

            ##############
            # LPIPS LOSS #
            ##############

            lpips_loss = lpips_loss_fn(predicted_image, actual_image).mean()


            ##############
            # TOTAL LOSS #
            ##############

            loss = (lambda_diffuion * diffusion_mse_loss) + (lambda_clip * clip_loss) + (lambda_lpips * lpips_loss)

            loss = loss / acc_steps
            loss.backward()
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        # flush the left over
        if step_count % acc_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg = (epoch_loss / len(dataloader)) * acc_steps
        losses.append(avg)

        print(f"[DiT] Epoch {epoch+1}/{epochs}  loss={avg:.6f}")
        
        if (epoch % config["saves"]["DiT_Save_every"] == 0) or (epoch == epochs-1):
            torch.save(dit.state_dict(), config["saves"]["DiT_Path"])