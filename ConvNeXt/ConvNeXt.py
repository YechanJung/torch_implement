import torch
from torch import nn
from torchvision.ops.misc import Permute
from torchvision.ops import StochasticDepth

class CNBlock(nn.Module):
    def __init__(self, in_channels, layer_scale, stochastic_depth_prob):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels),
                                      Permute([0, 2, 3, 1]),
                                      nn.LayerNorm(in_channels, eps=1e-6),
                                      Permute([0, 3, 1, 2]),
                                      nn.Conv2d(in_channels, 4 * in_channels, 1),
                                      nn.GELU(),
                                      nn.Conv2d(4 * in_channels, in_channels, 1))
        self.layer_scale = nn.Parameter(torch.ones(1,in_channels, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x):
        residual = self.layer_scale * self.residual(x)
        residual = self.stochastic_depth(residual)
        out = residual + x
        return out

class ConvNeXt(nn.Module):
    def __init__(self, block_setting, stochastic_depth_prob = 0.0, layer_scale = 1e-6, num_classes = 1000, **kwargs):
        super().__init__()

        layers = []
        layers += [nn.Sequential(nn.Conv2d(3, block_setting[0][0], kernel_size=4, stride=4),
                                 Permute([0, 2, 3, 1]),
                                 nn.LayerNorm(block_setting[0][0], eps=1e-6),
                                 Permute([0, 3, 1, 2]))]

        total_stage_blocks = sum([setting[2] for setting in block_setting])
        stage_block_id = 0
        for in_channels, out_channels, num_blocks in block_setting:
            stage = []
            for _ in range(num_blocks):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1)
                stage.append(CNBlock(in_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers += [nn.Sequential(*stage)]
            if out_channels is not None:
                downsample = nn.Sequential(Permute([0, 2, 3, 1]),
                                           nn.LayerNorm(in_channels),
                                           Permute([0, 3, 1, 2]),
                                           nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2))
                layers += [downsample]

        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(block_setting[-1][0]),
                                        nn.Linear(block_setting[-1][0], num_classes))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

def ConvNeXt_T(**kwargs):
    block_setting = [[96, 192, 3],
                     [192, 384, 3],
                     [384, 768, 9],
                     [768, None, 3]]
    return ConvNeXt(block_setting, stochastic_depth_prob = 0.1,  **kwargs)

def ConvNeXt_S(**kwargs):
    block_setting = [[96, 192, 3],
                     [192, 384, 3],
                     [384, 768, 27],
                     [768, None, 3]]
    return ConvNeXt(block_setting, stochastic_depth_prob = 0.4, **kwargs)

def ConvNeXt_B(**kwargs):
    block_setting = [[128, 256, 3],
                     [256, 512, 3],
                     [512, 1024, 27],
                     [1024, None, 3]]
    return ConvNeXt(block_setting, stochastic_depth_prob = 0.5, **kwargs)

def ConvNeXt_L(**kwargs):
    block_setting = [[192, 384, 3],
                     [384, 768, 3],
                     [768, 1536, 27],
                     [1536, None, 3]]
    return ConvNeXt(block_setting, stochastic_depth_prob = 0.5, **kwargs)