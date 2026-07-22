
import torch
from einops import rearrange
from typing import Tuple, List



def get_time_embedding(time_steps, temb_dim):

    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=temb_dim // 2,
        dtype=torch.float32,
        device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps.unsqueeze(-1).repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


def get_patch_position_embedding(pos_emb_dim, grid_size: Tuple, device):
    assert pos_emb_dim % 4 == 0, 'Position embedding dimension must be divisible by 4'

    grid_size_h, grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device = device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32, device = device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0) # [2, grid_size[0], grid_size[1]]

    # grid_h_positions -> (Number of patch tokens,)
    grid_h_positions = grid[0].reshape(-1) # grid_size[0] * grid_size[1]
    grid_w_positions = grid[1].reshape(-1) # grid_size[0] * grid_size[1]

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=pos_emb_dim // 4,
        dtype=torch.float32,
        device=device) / (pos_emb_dim // 4))
    )

    grid_h_emb = grid_h_positions.unsqueeze(-1).repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)
    # grid_h_emb -> (Number of patch tokens, pos_emb_dim // 2)

    grid_w_emb = grid_w_positions.unsqueeze(-1).repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)

    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    # pos_emb -> (Number of patch tokens, pos_emb_dim)
    return pos_emb # [grid_size[0] * grid_size[1], pos_emb_dim]



class PatchEmbedding(torch.nn.Module):
    def __init__(self, grid_height, grid_width, g_channels, patch_height, patch_width, d_model):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.g_channels = g_channels

        self.d_model = d_model

        self.patch_height = patch_height
        self.patch_width = patch_width

        # Input dimension for Patch Embedding FC Layer
        patch_dim = self.g_channels * self.patch_height * self.patch_width

        self.patch_embed = torch.nn.Sequential(torch.nn.Linear(patch_dim, self.d_model))

        torch.nn.init.xavier_uniform_(self.patch_embed[0].weight)
        torch.nn.init.constant_(self.patch_embed[0].bias, 0)

    def forward(self, x):
        grid_size_h = self.grid_height // self.patch_height
        grid_size_w = self.grid_width // self.patch_width

        # B, C, H, W -> B, (Patches along height * Patches along width), Patch Dimension
        # Number of tokens = Patches along height * Patches along width
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                        ph=self.patch_height,
                        pw=self.patch_width)

        # BxNumber of tokens x Patch Dimension -> B x Number of tokens x Transformer Dimension
        out = self.patch_embed(out)

        # Add 2d sinusoidal position embeddings
        pos_embed = get_patch_position_embedding(pos_emb_dim=self.d_model, grid_size=(grid_size_h, grid_size_w), device=x.device)
        out += pos_embed
        return out # [B, grid_height/patch_height * grid_width/patch_width, d_model]