/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_KERNELS_PACKED_CONV_HPP
#define MIGRAPHX_GUARD_KERNELS_PACKED_CONV_HPP

#include <migraphx/kernels/types.hpp>

namespace migraphx {

// Each instruction selects one half from each packed dword and accumulates in FP32.
template <int A, int B>
__device__ inline float packed_conv_mad_mix(uint32_t a, uint32_t b, float sum)
{
#if defined(__gfx900__) || defined(__gfx90c__)
    asm("v_mad_mix_f32 %0, %1, %2, %0 op_sel:[%3,%4,0] op_sel_hi:[1,1,0]"
        : "+v"(sum)
        : "v"(a), "v"(b), "n"(A), "n"(B));
    return sum;
#else
    const auto x = __builtin_bit_cast(half2, a);
    const auto y = __builtin_bit_cast(half2, b);
    return fmaf(float(x[A]), float(y[B]), sum);
#endif
}

// All lanes participate; only the calling wave's LDS accesses are ordered.
// Wave-private tiles never require synchronization with another split wave.
__device__ inline void packed_conv_wave_barrier()
{
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    asm volatile("" ::: "memory");
}

// The first supported family is batch-one NCHW,3x3/pad1,32 output channels,10x10.
// Split partitions K among waves; cross-wave phases retain workgroup barriers.
template <int IC, int TM, int TN, int TK, int Split, bool Pointwise>
__device__ void packed_conv(const half* __restrict__ x,
                            const half* __restrict__ weights,
                            const half* __restrict__ bias,
                            const half* __restrict__ pointwise_weights,
                            const half* __restrict__ pointwise_bias,
                            half* __restrict__ output)
{
    constexpr int IH = 10, IW = 10, OC = 32, OH = IH, OW = IW;
    constexpr int N = OH * OW, K = IC * 9, NB = (N + TN - 1) / TN;
    static_assert(OW % 2 == 0 && K % 2 == 0, "Packed spatial/filter pairs are aligned");
    static_assert(TM % 16 == 0 && TN % 8 == 0 && TK % 2 == 0,
                  "Tiles cover 16 rows and four column pairs");
    static_assert(OC % TM == 0, "The validated grid has no channel tail");
    static_assert(K % (TK * Split) == 0, "All waves execute the same barriers");
    const int t = threadIdx.x % 64, split = threadIdx.x / 64, mr = t / 4, nr = t % 4;
    const int mb = TM == OC ? 0 : blockIdx.x / NB * TM, nb = blockIdx.x % NB * TN;
    union arena
    {
        struct operands
        {
            uint32_t a[Split][TM][TK / 2 + 1];
            uint32_t b[Split][TK][TN / 2 + 1];
        } tile;
        float partial[Split > 1 ? Split - 1 : 1][TM][TN + 1];
    };
    __shared__ arena storage;
    auto& a                        = storage.tile.a;
    auto& b                        = storage.tile.b;
    float sums[TM / 16][TN / 8][2] = {};
#pragma unroll 1
    for(int base = split * TK; base < K; base += TK * Split)
    {
        for(int i = t; i < TM * (TK / 2); i += 64)
        {
            const int m = mb + i / (TK / 2), k = base + i % (TK / 2) * 2;
            a[split][i / (TK / 2)][i % (TK / 2)] =
                reinterpret_cast<const uint32_t*>(weights)[(m * K + k) / 2];
        }
        for(int i = t; i < TK * (TN / 2); i += 64)
        {
            const int k = base + i / (TN / 2), n = nb + i % (TN / 2) * 2, tap = k % 9;
            const int h = n / OW + tap / 3 - 1, col = n % OW + tap % 3 - 1;
            uint32_t low = 0, high = 0;
            if(n < N && h >= 0 && h < IH)
            {
                if(col >= 0 && col < IW)
                    low = __builtin_bit_cast(uint16_t, x[(k / 9 * IH + h) * IW + col]);
                if(col + 1 >= 0 && col + 1 < IW)
                    high = __builtin_bit_cast(uint16_t, x[(k / 9 * IH + h) * IW + col + 1]);
            }
            b[split][i / (TN / 2)][i % (TN / 2)] = low | (high << 16);
        }
        packed_conv_wave_barrier();
#pragma unroll
        for(int k = 0; k < TK / 2; ++k)
#pragma unroll
            for(int i = 0; i < TM / 16; ++i)
            {
                const uint32_t weight = a[split][mr + i * 16][k];
#pragma unroll
                for(int j = 0; j < TN / 8; ++j)
                {
                    const uint32_t low  = b[split][k * 2][nr + j * 4];
                    const uint32_t high = b[split][k * 2 + 1][nr + j * 4];
                    sums[i][j][0]       = packed_conv_mad_mix<0, 0>(weight, low, sums[i][j][0]);
                    sums[i][j][1]       = packed_conv_mad_mix<0, 1>(weight, low, sums[i][j][1]);
                    sums[i][j][0]       = packed_conv_mad_mix<1, 0>(weight, high, sums[i][j][0]);
                    sums[i][j][1]       = packed_conv_mad_mix<1, 1>(weight, high, sums[i][j][1]);
                }
            }
        packed_conv_wave_barrier();
    }
    auto& partial = storage.partial;
    if constexpr(Split > 1)
    {
        // Every wave must finish reading operands before any wave reuses the arena.
        __syncthreads();
        if(split != 0)
#pragma unroll
            for(int i = 0; i < TM / 16; ++i)
#pragma unroll
                for(int j = 0; j < TN / 8; ++j)
#pragma unroll
                    for(int p = 0; p < 2; ++p)
                        partial[split - 1][mr + i * 16][nr * 2 + j * 8 + p] = sums[i][j][p];
        __syncthreads();
    }
    static_assert(not Pointwise or (TM == OC and IC == 32),
                  "The pointwise head requires all 32 channels");
    __shared__ half activated[TN][TM] __attribute__((aligned(4)));
    // Only wave zero owns the epilogue after the cross-wave reduction.
    if(split != 0)
        return;
#pragma unroll
    for(int i = 0; i < TM / 16; ++i)
#pragma unroll
        for(int j = 0; j < TN / 8; ++j)
        {
            float value[2];
#pragma unroll
            for(int p = 0; p < 2; ++p)
            {
#pragma unroll
                for(int k = 1; k < Split; ++k)
                    sums[i][j][p] += partial[k - 1][mr + i * 16][nr * 2 + j * 8 + p];
                const int m = mb + mr + i * 16;
                value[p]    = sums[i][j][p] + float(bias[m]);
                value[p] /= 1.f + expf(-value[p]);
            }
            const int m = mb + mr + i * 16, n = nb + nr * 2 + j * 8;
            if constexpr(Pointwise)
            {
                if(n < N)
                {
                    activated[n - nb][m]     = half(value[0]);
                    activated[n - nb + 1][m] = half(value[1]);
                }
            }
            else if(n < N)
                reinterpret_cast<half2*>(output)[(m * N + n) / 2] =
                    half2{half(value[0]), half(value[1])};
        }

    if constexpr(Pointwise)
    {
        // Only wave zero produces and consumes activated; all its lanes participate.
        packed_conv_wave_barrier();
        for(int index = t; index < 4 * TN; index += 64)
        {
            const int channel = index / TN, col = index % TN, n = nb + col;
            if(n >= N)
                continue;
            float sum = 0;
#pragma unroll
            for(int c = 0; c < OC / 2; ++c)
            {
                const uint32_t packed_x = reinterpret_cast<const uint32_t*>(activated[col])[c];
                const uint32_t packed_w =
                    reinterpret_cast<const uint32_t*>(pointwise_weights)[channel * OC / 2 + c];
                sum = packed_conv_mad_mix<0, 0>(packed_x, packed_w, sum);
                sum = packed_conv_mad_mix<1, 1>(packed_x, packed_w, sum);
            }
            output[channel * N + n] = half(sum + float(pointwise_bias[channel]));
        }
    }
}

} // namespace migraphx
#endif
