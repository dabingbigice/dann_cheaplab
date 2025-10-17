import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
def patch_divide(x, step, ps):
    """Crop image into patches."""
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        down = min(i + ps, h)
        top = down - ps
        nh += 1
        for j in range(0, w + step - ps, step):
            right = min(j + ps, w)
            left = right - ps
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()
    return crop_x, nh, nw

def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image."""
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        down = min(i + ps, h)
        top = down - ps
        for j in range(0, w + step - ps, step):
            right = min(j + ps, w)
            left = right - ps
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    # 处理重叠区域的均值化
    for i in range(step, h + step - ps, step):
        top = i
        down = min(top + ps - step, h)
        if top < down:
            output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = min(left + ps - step, w)
        if left < right:
            output[:, :, :, left:right] /= 2
    return output

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super().__init__()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size,
                               padding=(kernel_size-1)//2, groups=hidden_features)
        self.gelu = nn.GELU()

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], -1, x_size[0], x_size[1])
        x = self.gelu(self.dwconv(x))
        return x.flatten(2).transpose(1, 2).contiguous()

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, kernel_size=5):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dwconv = dwconv(hidden_features, kernel_size)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        return self.fc2(x)

class LRSA(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, ps, heads=1):
        super().__init__()
        self.ps = ps
        self.attn = PreNorm(dim, Attention(dim, heads, qk_dim))
        self.ffn = PreNorm(dim, ConvFFN(dim, mlp_dim))

    def forward(self, x):
        step = self.ps - 2
        crop_x, nh, nw = patch_divide(x, step, self.ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
        # Attention
        attned = self.attn(crop_x) + crop_x
        # FFN
        attned = self.ffn(attned, x_size=(ph, pw)) + attned
        # Rebuild
        attned = rearrange(attned, '(b n) (h w) c -> b n c h w', n=n, h=ph)
        return patch_reverse(attned, x, step, self.ps)

if __name__ == "__main__":
    x = torch.randn(1, 32, 256, 256)
    model = LRSA(dim=32, qk_dim=36, mlp_dim=96, ps=16)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")