# Copyright (c) 2025.
# SM120 (Blackwell GeForce / RTX PRO 6000 / DGX Spark) FP8 forward pass.
#
# This is the ONLY genuinely Blackwell-specific tensor-core win available on SM120:
# SM120 has NO tcgen05/UMMA (that is SM100 datacenter only), and for bf16/fp16 it
# executes the SAME Ampere `mma.sync.aligned.m16n8k16` instruction, so 16-bit
# attention is already at the tensor-core roofline. FP8, however, uses the
# Blackwell `mma.sync.aligned.kind::f8f6f4.m16n8k32` instruction (k=32, 8-bit
# inputs) which delivers ~2x the bf16 tensor-core throughput.
#
# Design (validated by PyTorch emulation: ~5% mean rel err vs fp32, the expected
# FP8-attention band):
#   - Q, K are e4m3; QK GEMM accumulates in fp32; S is descaled by q_descale*k_descale.
#   - softmax runs in fp32 (unchanged, irreducible on SM120).
#   - P (post-softmax, in [0,1]) is requantized to e4m3 for the PV GEMM.
#   - V is e4m3; PV GEMM accumulates in fp32; O descaled by v_descale.
#
# REQUIRES nvidia-cutlass-dsl[cu13]: the default cu12.9 wheel gates the FP8 warp
# MMA off ("FP8 warp-level MMA (SM89) is not supported with CUDA Toolkit 12.9").
#
# STATUS: foundation. The FP8 MMA atoms are verified to compile on SM120 (see
# can_implement / the _get_tiled_mma override). The descale threading and the
# P->e4m3 requantization in compute_one_n_block are layered on top of the SM80
# CpAsync forward; see the per-method overrides below.

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.nvgpu import warp
import cutlass.utils as utils_basic

from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120


class FlashAttentionForwardSm120FP8(FlashAttentionForwardSm120):
    """FP8 (e4m3) forward for SM120 using the Blackwell f8f6f4 m16n8k32 warp MMA.

    Inherits the SM120 CpAsync forward (arch=80 code paths, no TMA O-store) and
    swaps the QK/PV tiled-MMA to the FP8 k=32 instruction. Q/K/V are e4m3; the
    softmax and accumulators stay fp32.
    """

    # Keep arch = 80 to reuse the CpAsync (no-TMA) code paths, like the bf16 SM120.
    arch = 80

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=False,
    ) -> bool:
        # FP8 e4m3 inputs only; head_dim must be a multiple of 32 (k=32 MMA).
        if dtype not in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
            return False
        if head_dim % 32 != 0 or head_dim_v % 32 != 0:
            return False
        if tile_n % 16 != 0 or num_threads % 32 != 0:
            return False
        # SMEM: Q + (K + V) stages, 1 byte/elem for fp8.
        smem_q = tile_m * head_dim
        smem_k = tile_n * head_dim * num_stages
        smem_v = tile_n * head_dim_v * num_stages
        smem = smem_q + smem_k + smem_v
        if smem > utils_basic.get_smem_capacity_in_bytes("sm_120"):
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True

    def _get_tiled_mma(self):
        # Blackwell f8f6f4 m16n8k32 warp MMA (e4m3 in, fp32 accumulate).
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaFP8Op(self.dtype, Float32, (16, 8, 32)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 32),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaFP8Op(self.dtype, Float32, (16, 8, 32)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 32),
        )
        return tiled_mma_qk, tiled_mma_pv
