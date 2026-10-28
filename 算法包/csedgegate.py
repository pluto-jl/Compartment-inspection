"""Candidate module; engineering checked, detection accuracy remains unverified."""
import torch
from torch import nn
from torch.nn import functional as F
 
 
def cba(cin, cout, kernel=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, padding=kernel // 2,
                  groups=groups, bias=False),
        nn.BatchNorm2d(cout), nn.SiLU())
 
 
class CSEdgeGate(nn.Module):
    """[P3, P4] -> enhanced P3. A1=(False,False), A2=(True,False)."""
    def __init__(self, c3, c4, d=32, eps=0.2,
                 use_edge=True, use_gate=True, gate_source='cross', gate_mode='sigmoid'):
        super().__init__()
        if min(c3, c4, d) < 1 or not 0 < eps < 1:
            raise ValueError('Invalid channels or gate lower bound')
        self.d = int(d)
        self.eps = float(eps)
        self.use_edge = bool(use_edge)
        self.use_gate = bool(use_gate)
        if gate_source not in ('cross', 'p3'):
            raise ValueError('gate_source must be cross or p3')
        self.gate_source = gate_source
        if gate_mode not in ('sigmoid', 'centered'):
            raise ValueError('gate_mode must be sigmoid or centered')
        self.gate_mode = gate_mode
        self.context_channels = int(c4)
        self.proj3 = cba(c3, d)
        self.texture = cba(d, d, 3, groups=d)
        self.fuse = cba(2 * d, d)
        self.out = nn.Sequential(
            nn.Conv2d(d, c3, 1, bias=False), nn.BatchNorm2d(c3))
        if self.use_gate:
            self.proj4 = cba(c4, d)
            self.gate = nn.Sequential(
                cba(2 * d, 2 * d, 3, groups=2 * d),
                nn.Conv2d(2 * d, 1, 1))
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.],
                           [-1., 0., 1.]]) / 8.0
        self.register_buffer('sobel_x', kx[None, None].repeat(d, 1, 1, 1))
        self.register_buffer('sobel_y', kx.T[None, None].repeat(d, 1, 1, 1))
        self.alpha = nn.Parameter(torch.tensor(0.1))
 
    def forward(self, x):
        x3, x4 = x
        v = self.proj3(x3)
        if self.use_edge:
            padded = F.pad(v, (1, 1, 1, 1), mode='replicate')
            gx = F.conv2d(padded, self.sobel_x, groups=self.d)
            gy = F.conv2d(padded, self.sobel_y, groups=self.d)
            edge = gx.abs() + gy.abs()
        else:
            edge = v  # same fusion width and trainable parameter count
        r = self.out(self.fuse(torch.cat((edge, self.texture(v)), dim=1)))
        if self.use_gate:
            context = x4
            if getattr(self, 'gate_source', 'cross') == 'p3':
                # Capacity-matched condition replacement. Only X4 dimensions are
                # consulted; no X4 values enter A4's conditioning branch.
                h3,w3=x3.shape[-2:];h4,w4=x4.shape[-2:]
                if h3 % h4 or w3 % w4:
                    raise ValueError('A4 requires integral P3/P4 spatial ratio')
                context = F.avg_pool2d(x3, (h3//h4,w3//w4))
                channels = self.context_channels
                context = context.repeat(1, (channels + x3.shape[1] - 1) // x3.shape[1], 1, 1)
                context = context[:, :channels]
            u = F.interpolate(self.proj4(context), size=v.shape[-2:],
                              mode='bilinear', align_corners=False)
            logits = self.gate(torch.cat((v, u), dim=1))
            if getattr(self, 'gate_mode', 'sigmoid') == 'centered':
                # A spatially constant map yields s=1, regardless of its bias.
                # The gate can redistribute the residual instead of uniformly
                # suppressing it to eps. Mean(s) need not equal one in general.
                contrast = logits - logits.mean(dim=(-2, -1), keepdim=True)
                s = 1.0 + (1.0 - self.eps) * torch.tanh(contrast)
            else:
                s = self.eps + (1.0 - self.eps) * torch.sigmoid(logits)
            r = r * s  # after output BN; no BN after this multiplication
        return x3 + self.alpha * r
