# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import math
import typing as tp
from typing import Union, Any, List
from einops import rearrange, pack, unpack

import torch
from sympy import false
from torch import nn

from . import local_vector_quantize_pytorch


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class VQEmbed(nn.Module):
    def __init__(self, vq, feature_dim, codebook_dim):
        super().__init__()
        self.vq = vq
        if feature_dim == codebook_dim:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()
        else:
            self.project_in = nn.Linear(feature_dim, codebook_dim)
            self.project_out = nn.Linear(codebook_dim, feature_dim)

    def to_indices(self, features):
        raise NotImplementedError

    def to_features(self, indices):
        latents = self.vq.indices_to_codes(indices)
        latents = rearrange(latents, 'b n d -> b d n')
        return latents

    def forward(self, x):
        latents = rearrange(x, 'b d n -> b n d')
        q_latents, indices, *ret = self.vq(latents)
        q_features = q_latents
        vq_loss = ret[0].sum() if len(ret) > 0 else torch.Tensor([0.]).to(dtype=x.dtype, device=x.device)
        return q_features, indices, vq_loss


def build_vq(name: str = "hexagon", feature_dim: int = 512, codebook_dim: Union[int, Any] = 8, codebook_num: int = 1):
    """
    Build a Q2D2 quantizer.
    Args:
        name (str): Grid type ("hexagon", "rectangle", "rhombic").
        feature_dim (int): Input feature dimension, shape (B, T, feature_dim).
        codebook_dim (int | list[int]): Codebook size or list of per-pair levels.
            Special cases: 4 → [7,7,7,7], 6 → [7,7,7,7,7,7].
        codebook_num (int): Must be 1 for supported grid types.
    Returns:
        VQEmbed: Maps input (B, T, feature_dim) → (B, T) or (B, T, codebook_num).
    """
    if name == "hexagon" or name == "rectangle" or name == "rhombic":
        assert codebook_num == 1
        if codebook_dim == 4:
            cb_levels = [7, 7, 7, 7]
        elif codebook_dim == 6:
            cb_levels = [7, 7, 7, 7, 7, 7]
        else:
            cb_levels = codebook_dim
        codebook_dim = len(cb_levels)
        print(f"grid type: {name}")
        print(f"codebook_levels: {cb_levels}")
        print(f"codebook dim: {codebook_dim}")
        print(f"feature dim: {feature_dim}")
        vq = local_vector_quantize_pytorch.Q2D(
            dim = feature_dim,
            vq_type = name,
            levels=cb_levels
        )
    else:
        raise ValueError(f"Unknown vq name: {name}")

    return VQEmbed(vq, feature_dim, codebook_dim)
    
    

class VectorQuantizer(nn.Module):
    """General vector quantizer module with support for residual (RVQ) and 2D grid quantization.
    Args:
        dimension (int): Input feature dimension.
        n_q (int): Number of residual quantizers (RVQ mode).
        bins (int): Codebook size (unused in Q2D2 grids).
        decay (float): EMA decay for codebook updates.
        kmeans_init (bool): Use k-means for codebook initialization.
        kmeans_iters (int): Iterations for k-means initialization.
        threshold_ema_dead_code (int): Minimum usage threshold before replacing codes.
        vq_type (str): Quantizer type ("rvq", "hexagon", "rectangle", "rhombic").
        codebook_dim (list[int]): Per-pair grid levels for Q2D2.
    """
    def __init__(
        self,
        dimension: int = 512,
        n_q: int = 1,
        bins: int = 1024, #not in use in Q2D2
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        vq_type: str = "hexagon",
        codebook_dim: List[int] = [7, 7, 7, 7, 7, 7],
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq_type = vq_type
        self.codebook_dim = codebook_dim


        if self.codebook_dim is None:
            self.codebook_dim = [7, 7, 7, 7, 7, 7]
        
        self.quantizer = build_vq(
            name = self.vq_type,
            feature_dim = self.dimension,
            codebook_dim = self.codebook_dim,
        )

    def forward(self, x: torch.Tensor, frame_rate: int, bandwidth: tp.Optional[float] = None) -> QuantizedResult:
        """Apply vector quantization during training.
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D).
            frame_rate (int): Input frame rate.
            bandwidth (float, optional): Target bandwidth.

        Returns:
            QuantizedResult: Quantized output, codes, bandwidth, and penalty loss.
        """

        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        nq_choice=[4,6,8]

        if self.training:
            choice = int(torch.randint(0, 3, (1,)).item())
            n_q=nq_choice[choice]
        if self.vq_type == 'rvq':
            quantized, codes, commit_loss = self.quantizer(x, n_q=n_q)
        else:
            quantized, codes, commit_loss = self.quantizer(x)
        bw = torch.full((x.shape[0],), n_q * bw_per_q).to(x.device)
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss))

    def infer(self, x: torch.Tensor, frame_rate: int, bandwidth: tp.Optional[float] = None) -> QuantizedResult:
        """Run vector quantization in inference mode (single quantizer).
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D).
            frame_rate (int): Input frame rate.
            bandwidth (float, optional): Target bandwidth.

        Returns:
            QuantizedResult: Quantized output, codes, bandwidth, and penalty loss.
        """
        n_q=1
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        if self.vq_type=='rvq':
            quantized, codes, commit_loss = self.quantizer(x, n_q=n_q)
        else:
            quantized, codes, commit_loss = self.quantizer(x)
        bw = torch.full((x.shape[0],), n_q * bw_per_q).to(x.device)
        return QuantizedResult(quantized, codes, bw, penalty = torch.mean(commit_loss))

    def get_num_quantizers_for_bandwidth(self, frame_rate: int, bandwidth: tp.Optional[float] = None) -> int:
        """Return n_q based on specified target bandwidth.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = self.n_q
        if bandwidth and bandwidth > 0.:
            n_q = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return n_q

    def get_bandwidth_per_quantizer(self, frame_rate: int):
        """Return bandwidth per quantizer for a given input frame rate.
        Each quantizer encodes a frame with lg(bins) bits.
        """
        return math.log2(self.bins) * frame_rate

    def encode(self, x: torch.Tensor, frame_rate: int, bandwidth: tp.Optional[float] = None) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizers to use
        and returns indices for each quantizer.
        """
        quantized, codes, commit_loss = self.quantizer(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        """
        quantized = self.quantizer.to_features(codes)
        return quantized
