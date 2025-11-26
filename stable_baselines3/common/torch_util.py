from typing import Dict, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init
from typing import Tuple


class FanInInitReLULayer(nn.Module):
    def __init__(
        self,
        inchan: int,
        outchan: int,
        layer_type: str = "conv",
        init_scale: float = 1.0,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation: bool = True,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        # Layer
        layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
        self.layer = layer(inchan, outchan, bias=self.norm is None, **layer_kwargs)
        self.use_activation = use_activation

        # Initialization
        self.layer.weight.data *= init_scale / self.layer.weight.norm(
            dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        )
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

    def forward(self, x: th.Tensor):
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x

class NormalizeEwma(nn.Module):
    def __init__(
        self,
        insize: int,
        norm_axes: int = 1,
        beta: float = 0.99,
        epsilon: float = 1e-2,
    ):
        super().__init__()

        # Params
        self.norm_axes = norm_axes
        self.beta = beta
        self.epsilon = epsilon

        # Parameters
        self.running_mean = nn.Parameter(th.zeros(insize), requires_grad=False)
        self.running_mean_sq = nn.Parameter(th.zeros(insize), requires_grad=False)
        self.debiasing_term = nn.Parameter(th.tensor(0.0), requires_grad=False)

    def running_mean_var(self) -> Tuple[th.Tensor, th.Tensor]:
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        var = (mean_sq - mean**2).clamp(min=1e-2)
        return mean, var

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.training:
            x_detach = x.detach()
            batch_mean = x_detach.mean(dim=tuple(range(self.norm_axes)))
            batch_mean_sq = (x_detach**2).mean(dim=tuple(range(self.norm_axes)))

            weight = self.beta
            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_mean_sq * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        mean, var = self.running_mean_var()
        mean = mean[(None,) * self.norm_axes]
        var = var[(None,) * self.norm_axes]
        x = (x - mean) / th.sqrt(var)
        return x

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        mean, var = self.running_mean_var()
        mean = mean[(None,) * self.norm_axes]
        var = var[(None,) * self.norm_axes]
        x = x * th.sqrt(var) + mean
        return x


class ScaledMSEHead(nn.Module):
    def __init__(
        self,
        insize: int,
        outsize: int,
        init_scale: float = 0.1,
        norm_kwargs: Dict = {},
    ):
        super().__init__()

        # Layer
        self.linear = nn.Linear(insize, outsize)

        # Initialization
        init.orthogonal_(self.linear.weight, gain=init_scale)
        init.constant_(self.linear.bias, val=0.0)

        # Normalizer
        self.normalizer = NormalizeEwma(outsize, **norm_kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(x)

    def normalize(self, x: th.Tensor) -> th.Tensor:
        return self.normalizer(x)

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        return self.normalizer.denormalize(x)

    def mse_loss(self, pred: th.Tensor, targ: th.Tensor) -> th.Tensor:
        targ = targ.view(-1, 1)
        targ_n = self.normalizer(targ).squeeze(-1)
        return F.mse_loss(pred, targ_n, reduction="none")
