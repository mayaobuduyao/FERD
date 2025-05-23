import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):  # Encode time step into embeddings
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            h = h + time_emb.view(-1, time_emb.shape[1], 1, 1)
        h = self.block2(h)
        return h + self.res_conv(x)

class UNet(nn.Module):
    def __init__(self, dim=64, init_dim=None, out_dim=None, channels=3, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        
        init_dim = init_dim if init_dim is not None else dim
        self.init_conv = nn.Conv2d(channels, init_dim, 1, padding=0)
        
        # time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # down blocks
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                    ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                    nn.Conv2d(dim_out, dim_out, 4, 2, 1) if ind < len(in_out) - 1 else nn.Identity()
                ])
            )
            
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # up blocks
        self.ups = nn.ModuleList([])
        reversed_in_out = list(reversed(in_out))
        for ind, (dim_in, dim_out) in enumerate(reversed_in_out):
            self.ups.append(
                nn.ModuleList([
                    ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                    ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                    nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1) if ind < len(reversed_in_out) - 1 else nn.Identity()
                ])
            )

        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, channels, 1)
        )

    def forward(self, x, time):
    
        x = self.init_conv(x)
        t = self.time_mlp(time)
        
        h = x
        intermediates = []

        # down
        for resnet1, resnet2, downsample in self.downs:
            h = resnet1(h, t)
            h = resnet2(h, t)
            intermediates.append(h)
            h = downsample(h)

        # middle
        h = self.mid_block1(h, t)
        h = self.mid_block2(h, t)

        # up
        for resnet1, resnet2, upsample in self.ups:
            h = torch.cat((h, intermediates.pop()), dim=1)
            h = resnet1(h, t)
            h = resnet2(h, t)
            h = upsample(h)

        return self.final_conv(h)


class DMGenerator(nn.Module):
    def __init__(self, unet_model, num_timesteps=50, img_size=32, device="cuda"):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.unet = unet_model.to(device)
        
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def forward(self, z):
        batch_size = z.shape[0]
        
        x = z.view(batch_size, -1, 1, 1)
        x = x.repeat(1, 1, self.img_size, self.img_size)
        x = x[:,:3,:,:]
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.ones(batch_size, device=self.device).long() * t  # construct time tensor

            predicted_noise = self.unet(x, t_batch.float() / self.num_timesteps)
            
            alpha = self.alphas[t]
            alpha_prev = self.alphas_cumprod[t-1] if t > 0 else torch.ones(1).to(self.device)
            sigma = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_prev)) * predicted_noise) + sigma * noise
            else:
                x = (1 / torch.sqrt(alpha)) * (x - 0 * predicted_noise)
            
        return x
