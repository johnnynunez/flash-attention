/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"

#include "static_switch.h"
#include "flash.h"
#include "tile_size.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd_sm90_tma.hpp"


using namespace cute;

template <int kHeadDim, int kBlockM, int kBlockN, int kStages, int ClusterM, typename Element, typename ElementOut,
          bool Is_causal, bool Is_local, bool Has_softcap, bool Varlen, bool PagedKV, bool AppendKV, bool PackGQA, bool Split, bool V_colmajor>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!(Is_causal && Is_local), "Causal and Local cannot be enabled at the same time");
    static_assert(!(AppendKV && V_colmajor), "AppendKV and V_colmajor cannot be enabled at the same time");
    static_assert(!(AppendKV && !Varlen), "AppendKV requires Varlen");
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;
    static constexpr bool FP8_TransposeV = Is_FP8 && !V_colmajor;
    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<Int<ClusterM>, _1, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopFwd<kStages, ClusterShape, TileShape_MNK, Element, float, cutlass::arch::Sm90, Is_causal, Is_local, Has_softcap, Varlen, PagedKV, AppendKV, PackGQA, Split, V_colmajor>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<TileShape_MNK, ClusterShape, ElementOut, CollectiveMainloop::NumMmaThreads, Varlen, PackGQA, FP8_TransposeV>;

    using SchedulerPersistent = std::conditional_t<Varlen,
        flash::VarlenDynamicPersistentTileScheduler<kBlockM, CollectiveMainloop::NumMmaThreads, CollectiveMainloop::NumProducerThreads, Split, PackGQA>,
        std::conditional_t<!Is_causal && !Is_local,
            flash::StaticPersistentTileScheduler<Split>,
            flash::DynamicPersistentTileScheduler<CollectiveMainloop::NumMmaThreads, CollectiveMainloop::NumProducerThreads, Split>>
    >;
    using SchedulerSingleTile = flash::SingleTileScheduler<Varlen, Split, PackGQA, kBlockM>;
    // If Split, PagedKV, or AppendKV then we probably don't have enough work for PersistentScheduler to be useful.
    using Scheduler = std::conditional_t<Split || PagedKV || AppendKV, SchedulerSingleTile, SchedulerPersistent>;
    using AttnKernel = flash::FlashAttnFwd<CollectiveMainloop, CollectiveEpilogue, Scheduler>;

    bool const is_varlen_q = params.cu_seqlens_q;
    bool const is_varlen_k = params.cu_seqlens_k;
    bool const is_varlen_k_new = params.cu_seqlens_knew;
    int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;
    int batch_q = !is_varlen_q ? params.b : 1;
    int batch_k = !is_varlen_k ? (params.kv_batch_idx ? params.b_k : params.b) : 1;
    typename CollectiveMainloop::StrideV v_strides =
        cute::conditional_return<!V_colmajor>(
            make_stride(params.v_row_stride, _1{}, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0),
            make_stride(_1{}, params.v_dim_stride, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0));
    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0},  // stride_Q
        static_cast<Element*>(params.k_ptr),
        {!PagedKV ? (!is_varlen_k ? params.seqlen_k : params.total_k) : params.page_size,
         params.d, params.h_k, !PagedKV ? batch_k : params.num_pages},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},  // stride_K
        static_cast<Element*>(params.v_ptr),
        v_strides,  // stride_V
        static_cast<Element const*>(params.knew_ptr),
        {!is_varlen_k_new ? params.seqlen_knew : params.total_knew, params.d, params.h_k, !is_varlen_k_new ? params.b : 1},  // shape_K_new
        {params.knew_row_stride, _1{}, params.knew_head_stride, !is_varlen_k_new ? params.knew_batch_stride : 0},  // stride_K_new
        static_cast<Element const*>(params.vnew_ptr),
        {params.vnew_row_stride, _1{}, params.vnew_head_stride, !is_varlen_k_new ? params.vnew_batch_stride : 0}, // stride_V_new
        static_cast<Element const*>(params.rotary_cos_ptr),
        {params.seqlen_k, params.rotary_dim / 2},  // shape_rotary, the seqlen shape doesn't matter
        {params.rotary_dim / 2, _1{}},  // stride_rotary_cos
        static_cast<Element const*>(params.rotary_sin_ptr),
        {params.rotary_dim / 2, _1{}},  // stride_rotary_sin
        params.is_rotary_interleaved,
        params.page_table,
        // if page_size is not set, avoid dividing by zero
        {params.kv_batch_idx ? params.b_k : params.b, !PagedKV ? 0 : params.seqlen_k / params.page_size}, // shape_page_table
        {params.page_table_batch_stride, _1{}},  // stride_page_table
        params.scale_softmax,
        params.q_descale_ptr, params.k_descale_ptr, params.v_descale_ptr,
        params.window_size_left, params.window_size_right, params.sink_token_length,
        params.softcap,
        params.num_splits,
        params.kv_batch_idx,
        params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
        params.seqused_q, params.seqused_k,
        params.leftpad_k,
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(!Split ? params.o_ptr : params.oaccum_ptr),
        {seqlen_q, params.d, params.h, batch_q, params.num_splits},  // shape_O
        {!Split ? params.o_row_stride : params.oaccum_row_stride,
         _1{},
         !Split ? params.o_head_stride : params.oaccum_head_stride,
         !is_varlen_q ? (!Split ? params.o_batch_stride : params.oaccum_batch_stride) : 0,
         !Split ? 0 : params.oaccum_split_stride},  // stride_O
        static_cast<float*>(!Split ? params.softmax_lse_ptr : params.softmax_lseaccum_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, !Split ? 0 : params.h * seqlen_q * batch_q},  // stride_LSE
        params.h_k,
        params.cu_seqlens_q, params.seqused_q
    };

    int qhead_per_khead = !PackGQA ? 1 : cutlass::ceil_div(params.h, params.h_k);
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));
    num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_m, !PackGQA ? params.h : params.h_k, params.b, params.num_splits,
        params.h / params.h_k,
        params.seqlen_q,
        params.tile_count_semaphore, params.cu_seqlens_q, params.seqused_q
    };

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    // int smem_size_q = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_q));
    // int smem_size_k = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_v));
    // printf("smem_size = %d, q = %d, k = %d, v = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v);
    // Get the ptr to kernel function.
    if constexpr (size(ClusterShape{}) > 1) {
        void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
        cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
    } else {
        auto kernel = cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
    }
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, int kBlockM, int kBlockN, int kHeadDim, int kStages,
        bool Is_causal, bool Is_local, bool Split, bool V_colmajor, bool Enable_cluster>
void run_mha_fwd_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    auto should_pack_gqa = [](int seqlen_q, int qhead_per_khead, int blockM) {
        // Heuristic: PackGQA is a bit slower but can help if seqlen_q is small or not near a multiple of kBlockM
        float nopack_gqa_efficiency = float(seqlen_q) / float(cute::round_up(seqlen_q, blockM));
        float pack_gqa_efficiency = float(seqlen_q * qhead_per_khead) / float(cute::round_up(seqlen_q * qhead_per_khead, blockM));
        return nopack_gqa_efficiency < 0.95 * pack_gqa_efficiency;
    };
    static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Split, std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>, float>;
    BOOL_SWITCH(params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k, Varlen, [&] {
        bool pack_gqa = params.pack_gqa >= 0  // if negative, we use a heuristic to decide
            ? bool(params.pack_gqa)
            // If varlen, we don't actually know seqlen_q but only max_seqlen_q.
            // If causal, PackGQA always seems faster
            : params.h != params.h_k && (Varlen || Is_causal || should_pack_gqa(params.seqlen_q, params.h / params.h_k, kBlockM));
        BOOL_SWITCH(params.page_table, PagedKV, [&] {
            BOOL_SWITCH(params.knew_ptr, AppendKV, [&] {
                BOOL_SWITCH(pack_gqa, PackGQA, [&] {
                //     BOOL_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                //         // Only use Cluster if number of tiles along seqlen_q is even and not varlen
                //         BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q * (!PackGQA ? 1 : params.h / params.h_k), kBlockM) % 2 == 0, UseCluster, [&] {
                            // run_flash_fwd<kHeadDim, kBlockM, kBlockN, kStages, !Is_causal && !Is_local && !Varlen && !Split && Enable_cluster && UseCluster ? 2 : 1, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PackGQA /*PackGQA*/, Split /*Split*/, false /*V_colmajor*/>(params, stream);
                            // run_flash_fwd<kHeadDim, kBlockM, kBlockN, kStages, !Is_causal && !Is_local && !Varlen && !Split && Enable_cluster && UseCluster ? 2 : 1, T, T_out, Is_causal, false, false, false /*Varlen*/, true /*PagedKV*/, false /*PackGQA*/, false /*Split*/, false /*V_colmajor*/>(params, stream);
                    run_flash_fwd<kHeadDim, kBlockM, kBlockN, kStages, 1, T, T_out, Is_causal, false, false, Varlen /*Varlen*/, PagedKV /*PagedKV*/, AppendKV && Varlen /*AppendKV*/, PackGQA /*PackGQA*/, Split /*Split*/, false /*V_colmajor*/>(params, stream);
                //         });
                //     });
                });
            });
        });
    });
}

template<typename T, int kHeadDim, bool Split>
void run_mha_fwd_hdim_16b(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        // Can't use structured binding since it's not compatible with constexpr
        static constexpr std::tuple<int, int> kBlock_MN = tile_size_fwd(kHeadDim, Is_causal || Is_local, sizeof(T) /*element_size*/);
        static constexpr bool Enable_cluster = kHeadDim >= 192 && !Is_causal && !Is_local && !Split;
        run_mha_fwd_dispatch<T, std::get<0>(kBlock_MN), std::get<1>(kBlock_MN), kHeadDim, 2, Is_causal, Is_local, Split, false /*V_colmajor*/, Enable_cluster>(params, stream);
    });
}

template<typename T, bool Split>
void run_mha_fwd_fp8_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    // CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
    //     BOOL_SWITCH(params.v_dim_stride != 1, V_colmajor, [&] {
    //         run_mha_fwd_dispatch<T, 192, 160, 64, 3, Is_causal, Is_local, Split, V_colmajor, false /*Enable_cluster*/>(params, stream);
    //     });
    // });
}

template<typename T, bool Split>
void run_mha_fwd_fp8_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    // CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
    //     BOOL_SWITCH(params.v_dim_stride != 1, V_colmajor, [&] {
    //         run_mha_fwd_dispatch<T, 192, 128, 96, 3, Is_causal, Is_local, Split, V_colmajor, false /*Enable_cluster*/>(params, stream);
    //     });
    // });
}


template<typename T, bool Split>
void run_mha_fwd_fp8_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    // CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
    //     BOOL_SWITCH(params.v_dim_stride != 1, V_colmajor, [&] {
    //         run_mha_fwd_dispatch<T, 128, V_colmajor ? 192 : 224, 128, 2, Is_causal, Is_local, Split, V_colmajor, false /*Enable_cluster*/>(params, stream);
    //     });
    // });
}

template<typename T, bool Split>
void run_mha_fwd_fp8_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    // CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
    //     BOOL_SWITCH(params.v_dim_stride != 1, V_colmajor, [&] {
    //         run_mha_fwd_dispatch<T, 128, 160, 192, 2, Is_causal, Is_local, Split, V_colmajor, true /*Enable_cluster*/>(params, stream);
    //     });
    // });
}

template<typename T, bool Split>
void run_mha_fwd_fp8_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    // CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
    //     BOOL_SWITCH(params.v_dim_stride != 1, V_colmajor, [&] {
    //         run_mha_fwd_dispatch<T, 128, 128, 256, 2, Is_causal, Is_local, Split, V_colmajor, true /*Enable_cluster*/>(params, stream);
    //     });
    // });
}

