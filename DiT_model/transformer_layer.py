import torch

class TransformerLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model

        ff_hidden_dim = 4 * self.d_model

        # Layer norm for attention block
        self.att_norm = torch.nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1E-6)

        self.attn_block = torch.nn.MultiheadAttention(embed_dim = d_model, num_heads = num_heads, batch_first = True)
        
        # Layer norm for mlp block
        self.ff_norm = torch.nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1E-6)

        self.mlp_block = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, ff_hidden_dim),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Linear(ff_hidden_dim, self.d_model),
        )

        # Scale Shift Parameter predictions for this layer
        # 1. Scale and shift parameters for layernorm of attention (2 * d_model)
        # 2. Scale and shift parameters for layernorm of mlp (2 * d_model)
        # 3. Scale for output of attention prior to residual connection (d_model)
        # 4. Scale for output of mlp prior to residual connection (d_model)
        # Total 6 * d_model
        self.adaptive_norm_layer = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(self.d_model, 6 * self.d_model, bias=True)
        )

        torch.nn.init.xavier_uniform_(self.mlp_block[0].weight)
        torch.nn.init.constant_(self.mlp_block[0].bias, 0)
        torch.nn.init.xavier_uniform_(self.mlp_block[-1].weight)
        torch.nn.init.constant_(self.mlp_block[-1].bias, 0)

        torch.nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        torch.nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

    def forward(self, x, condition):
        scale_shift_params = self.adaptive_norm_layer(condition).chunk(6, dim=1)

        pre_attn_shift, pre_attn_scale, post_attn_scale, pre_mlp_shift, pre_mlp_scale, post_mlp_scale = scale_shift_params
        
        out = x

        attn_norm_output = (self.att_norm(out) * (1 + pre_attn_scale.unsqueeze(1)) + pre_attn_shift.unsqueeze(1))

        attn_out, _ = self.attn_block(
                        attn_norm_output,
                        attn_norm_output,
                        attn_norm_output
                    )

        out = out + post_attn_scale.unsqueeze(1) * attn_out

        mlp_norm_output = (self.ff_norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1))

        out = out + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm_output)

        return out