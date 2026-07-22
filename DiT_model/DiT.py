import torch
from typing import Tuple, List
from einops import rearrange
import json
from .transformer_layer import TransformerLayer
from .embeddings import *
    

    
class DiT(torch.nn.Module):
    def __init__(self, d_model, patch_size, grid_size, g_channels, CLIP_dim, timestep_emb_dim, number_emb_dim, num_layers, num_heads):
        super().__init__()

        num_layers       = num_layers
        self.grid_height = grid_size
        self.grid_width  = grid_size

        self.g_channels = g_channels
        
        self.d_model = d_model

        self.patch_height = patch_size
        self.patch_width  = patch_size

        self.timestep_emb_dim = timestep_emb_dim
        self.number_emb_dim = number_emb_dim

        self.CLIP_dim = CLIP_dim

        # Number of patches along height and width
        self.nh = self.grid_height // self.patch_height
        self.nw = self.grid_width // self.patch_width

        # Clip_dim -> model_dim

        self.clip_proj = torch.nn.Sequential(
            torch.nn.Linear(self.CLIP_dim, self.d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(self.d_model, self.d_model)
        )

        # Patch Embedding Block
        self.patch_embed_layer = PatchEmbedding(grid_height  = self.grid_height, 
                                                grid_width   = self.grid_width, 
                                                g_channels   = self.g_channels, 
                                                patch_height = self.patch_height, 
                                                patch_width  = self.patch_width,
                                                d_model      = self.d_model)
        

        # Initial projection from sinusoidal time embedding
        self.t_proj = torch.nn.Sequential(
            torch.nn.Linear(self.timestep_emb_dim, self.d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(self.d_model, self.d_model)
        )


        # All Transformer Layers
        self.layers = torch.nn.ModuleList([
            TransformerLayer(d_model = d_model, num_heads = num_heads) for _ in range(num_layers)
        ])

        # Final normalization for unpatchify block
        self.norm = torch.nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1E-6)

        # Scale and Shift parameters for the norm
        self.adaptive_norm_layer = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(self.d_model, 2 * self.d_model, bias=True)
        )

        # Final Linear Layer
        self.proj_out = torch.nn.Linear(self.d_model, self.patch_height * self.patch_width * self.g_channels)

        torch.nn.init.normal_(self.t_proj[0].weight, std=0.02)
        torch.nn.init.normal_(self.t_proj[2].weight, std=0.02)

        torch.nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        torch.nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

        torch.nn.init.constant_(self.proj_out.weight, 0)
        torch.nn.init.constant_(self.proj_out.bias, 0)

        torch.nn.init.normal_(self.clip_proj[0].weight, std=0.02)
        torch.nn.init.normal_(self.clip_proj[2].weight, std=0.02)

        self.g_channel_nn = torch.nn.Sequential(torch.nn.Linear(2*self.g_channels, self.g_channels))

    def forward(self, x, t, y, clip_emb = None):

        out = torch.cat([x,y], dim = 1) # [B, 8, 32, 32]

        B = out.shape[0]
        out = out.permute(0, 2, 3, 1) # [B, 32, 32, 8]
        out = self.g_channel_nn(out)  # [B, 32, 32, 4]
        out = out.permute(0, 3, 1, 2) # [B, 4, 32, 32]

        # Patchify
        out = self.patch_embed_layer(out)

        # Compute Timestep representation
        # t_emb -> (Batch, timestep_emb_dim)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.timestep_emb_dim)

        # (Batch, timestep_emb_dim) -> (Batch, d_model)
        c_emb = self.t_proj(t_emb)

        if clip_emb is not None:
            c_emb = c_emb + self.clip_proj(clip_emb)

        # Go through the transformer layers
        for layer in self.layers:
            out = layer(out, c_emb)

        # Shift and scale predictions for output normalization
        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_layer(c_emb).chunk(2, dim=1)
        out = (self.norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
               pre_mlp_shift.unsqueeze(1))

        # Unpatchify
        # (B,patches,d_model) -> (B,patches,channels * patch_width * patch_height)
        out = self.proj_out(out)
        out = rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                        ph=self.patch_height,
                        pw=self.patch_width,
                        nw=self.nw,
                        nh=self.nh)
        return out
    


if __name__ == "__main__":
    
    dit = DiT(d_model          = 768,
              patch_size       = 2,
              grid_size        = 28,
              g_channels       = 4,
              timestep_emb_dim = 768,
              number_emb_dim   = 512,
              num_layers       = 12,
              num_heads        = 16,
              CLIP_dim         = 768)
    
    print(f"{sum(i.numel() for i in dit.parameters()):,}")

    B = 2

    x = torch.rand(B, 4, 28, 28)
    y = torch.rand(B, 4, 28, 28)
    t = torch.arange(0, B, 1)

    clip_out = torch.rand(B, 768)

    model_out = dit(x, t, y, clip_out)
    print(model_out.shape)