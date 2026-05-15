"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor, int32
from torch.amp import autocast

import einx
from einops import rearrange, pack, unpack

import random

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_two(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_two

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z):
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

def ste(z, bounded_z):
    """straight through gradients."""
    bounded_z = z + (bounded_z - z).detach()
    return bounded_z

def floor_ste(z):
    zhat = z.floor()
    return z + (zhat - z).detach()

def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)

def safe_div(num, den, eps = 1e-6):
    return num / den.clamp(min = eps)

# main class
class Q2D(Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices = True,
        force_quantization_f32 = True,
        preserve_symmetry: bool = False,
        noise_dropout = 0.0,
        vq_type = "hexagon",
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale

        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim
        self.num_pairs = self.codebook_dim // 2

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Sequential(nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias),nn.Tanh())  # This bounds the output to [-1, 1]
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections

        #grid quantization type
        self.vq_type = vq_type

        self.return_indices = return_indices

        if return_indices:

            assert self.codebook_dim % 2 == 0, "2D grid requires even number of features (pairs)"
            self.num_pairs = self.codebook_dim // 2
            device = self._levels.device

            def build_grid():
                """
                Build grids for all 2D channel pairs.
                Depending on `vq_type`, constructs per-pair tilings:
                    - hexagonal
                    - rectangular
                    - rhombic
                Returns:
                    grids: List of per-pair grid tensors.
                    grid_lens: List of grid sizes per pair.
                """
                grids = []
                grid_lens = []
                for i in range(self.num_pairs):
                    #levels = self._levels[2 * i].item()  # per pair level
                    if self.vq_type == "hexagon":
                        levels = self._levels[2 * i].item()  # per pair level
                        grid = self.generate_hex_grid(levels=levels, device=device, extent=((levels-1)/2))
                    elif self.vq_type == "rectangle":
                        x_levels = self._levels[2*i].item()
                        y_levels = self._levels[(2*i)+1].item()  # per pair level
                        grid = self.generate_rectangle_grid(x_levels=x_levels, y_levels=y_levels, device=device, x_extent=((x_levels-1)/2), y_extent=((y_levels-1)/2))
                    elif self.vq_type == "rhombic":
                        x_levels = self._levels[2*i].item()
                        y_levels = self._levels[(2*i)+1].item()  # per pair level
                        grid = self.generate_rhombic_grid(x_levels=x_levels, y_levels=y_levels, device=device, x_extent=((x_levels-1)/2), y_extent=((y_levels-1)/2))
                    else:
                        raise ValueError(f"Unsupported vq_type: {self.vq_type}")
                    grids.append(grid)
                    grid_lens.append(len(grid))
                return grids, grid_lens

            # Build and register
            grids, grid_lens = build_grid()
            self.tile_grid = grids  # list of [G_i, 2] tensors
            self.register_buffer("grid_len", torch.tensor(grid_lens, device=device), persistent=False)  # [P]

            # Compute grid_basis: [1, grid_len[0], grid_len[0]*grid_len[1], ...]
            grid_basis = torch.ones(self.num_pairs, dtype=torch.long, device=device)
            for i in range(1, self.num_pairs):
                grid_basis[i] = grid_basis[i - 1] * grid_lens[i - 1]
                self.register_buffer("grid_basis", grid_basis, persistent=False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def generate_hex_grid(self, levels: int, device: torch.device, extent: float = 3.0) -> torch.Tensor:
        """
            Generate a hexagonal tiling grid.
            Uses alternating row offsets to simulate hexagonal packing.
            Args:
                levels: Number of grid steps along each axis.
                extent: Range of the grid (symmetric around 0).
            Returns:
                grid: Tensor of shape [G, 2] with hexagonal coordinates.
        """
        assert levels >= 2, "levels should be >= 2 to have center points"

        dx = 2 * extent / (levels - 1)
        dy = dx * (3 ** 0.5) / 2

        y_coords = torch.linspace(-extent, extent, levels, device=device)
        grid_points = []

        for i, y in enumerate(y_coords):
            x_offset = (-dx / 4) if i % 2 else (dx / 4)
            x_coords = torch.linspace(-extent, extent, levels, device=device) + x_offset
            grid = torch.stack(torch.meshgrid(x_coords, y[None], indexing="ij"), dim=-1).reshape(-1, 2)
            grid_points.append(grid)

        return torch.cat(grid_points, dim=0)  # [G, 2]

    def generate_rectangle_grid(self, x_levels: int, y_levels: int, device: torch.device, x_extent: float = 3.0, y_extent: float = 3.0):
        """
           Generate a standard 2D rectangular grid.
           Args:
               x_levels, y_levels: Number of points along x and y.
               x_extent, y_extent: Range extents in each axis.
           Returns:
               grid: Tensor of shape [x_levels * y_levels, 2] with (x, y) coordinates.
        """
        assert x_levels >= 2, "levels should be >= 2 to have center points"
        assert y_levels >= 2, "levels should be >= 2 to have center points"
        
        x_coords = torch.linspace(-x_extent, x_extent, x_levels, device=device)
        y_coords = torch.linspace(-y_extent, y_extent, y_levels, device=device)
        grid = torch.stack(torch.meshgrid(x_coords, y_coords, indexing="ij"), dim=-1).reshape(-1, 2)
        return grid

    def generate_rhombic_grid(self, x_levels: int, y_levels: int, device: torch.device, x_extent: float = 3.0, y_extent: float = 3.0) -> torch.Tensor:
        """
            Generate a rhombic tiling grid in 2D.
            Combines:
            - Regular rectangular grid points.
            - Midpoints of each rectangle, producing a rhombic pattern.
            Returns:
                full_grid: Tensor of shape [G, 2] with rhombic lattice coordinates.
        """
        assert (x_levels % 2 == 1) and (y_levels % 2 == 1), "Need odd levels in both axes to guarantee (0,0)"
        
        dx = 2 * x_extent / (x_levels - 1)
        dy = 2 * y_extent / (y_levels - 1)
        
        x_coords = torch.linspace(-x_extent, x_extent, x_levels, device=device)
        y_coords = torch.linspace(-y_extent, y_extent, y_levels, device=device)
        
        # rectangle grid points
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
        regular_points = torch.stack((x_grid, y_grid), dim=-1).reshape(-1, 2)

        # midpoints (center of each rectangle)
        mid_x_coords = x_coords[:-1] + dx / 2
        mid_y_coords = y_coords[:-1] + dy / 2
        mid_x, mid_y = torch.meshgrid(mid_x_coords, mid_y_coords, indexing="ij")
        midpoint_points = torch.stack((mid_x, mid_y), dim=-1).reshape(-1, 2)

        # Combine both
        full_grid = torch.cat([regular_points, midpoint_points], dim=0)
        return full_grid

    def quantize_to_grid(self, z_pairs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Snap 2D feature pairs to their nearest grid points.
        Args:
            z_pairs: Tensor of shape [..., P, 2] where each pair lies in continuous space.
        Returns:
            z_pairs_quant: Snapped pairs aligned to grid points [..., P, 2].
            nearest: Integer indices of nearest grid points [..., P].
        """
        # z_pairs: [*, P, 2]
        device = z_pairs.device
        *prefix, P, _ = z_pairs.shape
        z_pairs_flat = z_pairs.view(-1, P, 2)  # [B*N*C, P, 2]

        snapped = []
        nearest_all = []

        for i in range(self.num_pairs):
            grid_i = self.tile_grid[i].to(device)  # [G_i, 2]
            pair_i = z_pairs_flat[:, i]            # [B*N*C, 2]
            dists = torch.cdist(pair_i.unsqueeze(1), grid_i.unsqueeze(0))  # [B*N*C, 1, G_i]
            nearest = dists.argmin(dim=-1)         # [B*N*C]
            snapped_i = grid_i[nearest]            # [B*N*C, 2]
            snapped.append(snapped_i)
            nearest_all.append(nearest)

        # Stack results
        z_pairs_quant = torch.stack(snapped, dim=1)      # [B*N*C, P, 2]
        nearest = torch.stack(nearest_all, dim=1)        # [B*N*C, P]

        z_pairs_quant = z_pairs_quant.view(*prefix, P, 2)
        nearest = nearest.view(*prefix, P)

        return z_pairs_quant, nearest

    def bound(self, z, eps: float = 1e-3):
        """
        Bound normalized input `z` into the valid grid range.
        Formula: scale z by half the grid resolution (levels - 1)/2, with
        a small epsilon margin to avoid overflow.
        Args:
            z: Continuous input tensor (..., D).
            eps: Small slack to ensure stability at boundaries.
        Returns:
            res: Bounded tensor scaled into quantization range.
        """
        half_l = (self._levels - 1) * (1 + eps) / 2
        res = z * half_l
        return res

    # symmetry-preserving and noise-approximated quantization, section 3.2 in https://arxiv.org/abs/2411.19842
    def symmetry_preserving_bound(self, z):
        """
        QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1
        """
        levels_minus_1 = (self._levels - 1)
        scale = 2.0 / levels_minus_1
        bracket = (levels_minus_1 * (torch.tanh(z) + 1) / 2.0) + 0.5
        bracket = floor_ste(bracket)
        return scale * bracket - 1.0

    def quantize(self, z):
        """
        Quantize continuous input `z` into discrete codes on a structured 2D grid.
        Steps:
        1. Bound or normalize `z` (with optional symmetry-preserving bound).
        2. Split features into pairs and snap each pair to its nearest grid point
           (hexagonal, rectangular, or rhombic tiling).
        3. Optionally apply noise dropout during training to encourage robustness.
        4. Use STE (straight-through estimator) to pass gradients through quantization.
        Returns:
            z_codes: Quantized tensor, same shape as `z`.
            nearest: Indices of the nearest grid points for each pair.
        """

        shape, device, noise_dropout, preserve_symmetry, half_width = z.shape[0], z.device, self.noise_dropout, self.preserve_symmetry, (self._levels // 2)
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

        bounded_z = bound_fn(z)

        # --- Apply Quantization for 2D Pairs ---
        if bounded_z.shape[-1] % 2 != 0:
            raise ValueError("Feature dimension must be even for 2D tiling.")

        z_pairs = bounded_z.reshape(*bounded_z.shape[:-1], -1, 2)
        z_pairs_grid, nearest = self.quantize_to_grid(z_pairs)
        nearest = nearest.view(*z_pairs.shape[:-1])
        nearest = nearest.squeeze(-2)  # [B, N, P]
        bounded_z = z_pairs_grid.reshape_as(bounded_z)

        # determine where to add a random offset elementwise
        # if using noise dropout

        if self.training and noise_dropout > 0.:
            offset_mask = torch.bernoulli(torch.full_like(bounded_z, noise_dropout)).bool()
            offset = torch.rand_like(bounded_z) - 0.5
            bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)
            
        # standard STE to get gradients through VQ layer.
        z_codes = ste(z, bounded_z) / half_width
        
        return z_codes, nearest


    def _indices_to_codes(self, indices):
        """
        Convert integer grid indices back into their corresponding normalized 2D codes.
        Args:
            indices: Encoded indices [B, N, P] where P is the number of 2D pairs.
        Returns:
            codes: Continuous normalized codes [B, N, D], reconstructed from grid positions.
        """
        P = self.num_pairs
        D = P * 2
        device = indices.device

        nearest = (indices.unsqueeze(-1) // self.grid_basis) % self.grid_len  # [B, N, P]
        z_pairs = []

        for i in range(P):
            grid_i = self.tile_grid[i].to(device)          # [G_i, 2]
            nearest_i = nearest[..., i]                    # [B, N]
            flat_nearest = nearest_i.reshape(-1)           # [B*N]
            z_pair_flat = grid_i[flat_nearest]             # [B*N, 2]
            z_pair = z_pair_flat.view(*nearest_i.shape, 2) # [B, N, 2]
            z_pairs.append(z_pair)

        z_pairs = torch.stack(z_pairs, dim=-2)  # [B, N, P, 2]
        half_widths = (self._levels[::1] // 2).to(device)
        half_widths = half_widths.view(1, 1, P, 2)
        codes = z_pairs / half_widths

        return codes.view(*indices.shape, D)

    def codes_to_indices(self, zhat, nearest=None):
        """
        Map quantized codes (or nearest-grid assignments) to flattened integer indices.

        Args:
            zhat: Quantized representation (not directly used here).
            nearest: Per-pair nearest grid indices [B, N, P].

        Returns:
            indices: Flattened integer indices [B, N, 1] suitable for embeddings or codebook lookup.
        """
        indices = (nearest * self.grid_basis).sum(dim=-1).to(int32)  # [B, N]
        indices = indices.unsqueeze(-1)  # [B, N, 1]
        return indices

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        #change
        #if need_move_channel_last:
        #    z = rearrange(z, 'b d ... -> b ... d')
        #    z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes, nearest = self.quantize(z)

            # returning indices could be optional
            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes,nearest)

            codes = rearrange(codes, 'b n c d -> b n (c d)')

            codes = codes.to(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        # return quantized output and indices

        return out, indices
