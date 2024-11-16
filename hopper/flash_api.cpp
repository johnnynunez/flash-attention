/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"

// Copied from https://github.com/pytorch/pytorch/commit/7931eee5c5ebcdf468bff4d308510b03355cd909
// This is so that we can pass in torch.dtype as a parameter to the function.
// TODO: does it conflict if user compiles it with a recent version of Pytorch that has that commit?
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace pybind11::detail {

    template <>
    struct type_caster<at::ScalarType> {
    public:
        // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
        PYBIND11_TYPE_CASTER(at::ScalarType, _("torch.dtype"));
        // PYBIND11_TYPE_CASTER defines a member field called value. at::ScalarType
        // cannot be default-initialized, we provide this constructor to explicitly
        // initialize that field. The value doesn't matter as it will be overwritten
        // after a successful call to load.
        type_caster() : value(at::kFloat) {}
        bool load(handle src, bool) {
            PyObject* obj = src.ptr();
            if (THPDtype_Check(obj)) {
                value = reinterpret_cast<THPDtype*>(obj)->scalar_type;
                return true;
            }
            return false;
        }
        static handle cast(
                           const at::ScalarType& src,
                           return_value_policy /* policy */,
                           handle /* parent */) {
            return Py_NewRef(torch::getTHPDtype(src));
        }
    };

} // namespace pybind11::detail


#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap=0.f) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.v_dim_stride = v.stride(-1);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.o_batch_stride = out.stride(0);
    }
    if (cu_seqlens_k_d == nullptr) {
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_q = static_cast<int *>(seqused_q);
    params.seqused_k = static_cast<int *>(seqused_k);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.softcap = softcap;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !params.is_causal;

    // TODO: check this
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k - 1; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_q - 1; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap=0.f,
                      bool deterministic=false) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     seqused_q,
                     seqused_k,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap);

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    // HEADDIM_SWITCH(params.d, [&] {
    //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
    // });
    if (!params.is_e4m3) {
        if (params.is_bf16) {
            if (params.d <= 64) {
                run_mha_fwd_<cutlass::bfloat16_t, 64>(params, stream);
            } else if (params.d <= 96) {
                run_mha_fwd_<cutlass::bfloat16_t, 96>(params, stream);
            } else if (params.d <= 128) {
                run_mha_fwd_<cutlass::bfloat16_t, 128>(params, stream);
            } else if (params.d <= 192) {
                run_mha_fwd_<cutlass::bfloat16_t, 192>(params, stream);
            } else {
                run_mha_fwd_<cutlass::bfloat16_t, 256>(params, stream);
            }
        } else {
            if (params.d <= 64) {
                run_mha_fwd_<cutlass::half_t, 64>(params, stream);
            } else if (params.d <= 96) {
                run_mha_fwd_<cutlass::half_t, 96>(params, stream);
            } else if (params.d <= 128) {
                run_mha_fwd_<cutlass::half_t, 128>(params, stream);
            } else if (params.d <= 192) {
                run_mha_fwd_<cutlass::half_t, 192>(params, stream);
            } else {
                run_mha_fwd_<cutlass::half_t, 256>(params, stream);
            }
        }
    } else {
        if (params.d <= 64) {
            run_mha_fwd_<cutlass::float_e4m3_t, 64>(params, stream);
        } else if (params.d <= 96) {
            run_mha_fwd_<cutlass::float_e4m3_t, 96>(params, stream);
        } else if (params.d <= 128) {
            run_mha_fwd_<cutlass::float_e4m3_t, 128>(params, stream);
        } else if (params.d <= 192) {
            run_mha_fwd_<cutlass::float_e4m3_t, 192>(params, stream);
        } else {
            run_mha_fwd_<cutlass::float_e4m3_t, 256>(params, stream);
        }
    }
}

void run_mha_fwd_combine(Flash_fwd_params &params, cudaStream_t stream) {
    // If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    // so that kBlockM is smaller and we have more parallelism.
    if (params.is_fp32) {
        if (params.d <= 64) {
            run_mha_fwd_combine_<float, 64>(params, stream);
        } else if (params.d <= 128) {
            run_mha_fwd_combine_<float, 128>(params, stream);
        } else {
            run_mha_fwd_combine_<float, 256>(params, stream);
        }
    } else if (params.is_bf16) {
        if (params.d <= 64) {
            run_mha_fwd_combine_<cutlass::bfloat16_t, 64>(params, stream);
        } else if (params.d <= 128) {
            run_mha_fwd_combine_<cutlass::bfloat16_t, 128>(params, stream);
        } else {
            run_mha_fwd_combine_<cutlass::bfloat16_t, 256>(params, stream);
        }
    } else {
        if (params.d <= 64) {
            run_mha_fwd_combine_<cutlass::half_t, 64>(params, stream);
        } else if (params.d <= 128) {
            run_mha_fwd_combine_<cutlass::half_t, 128>(params, stream);
        } else {
            run_mha_fwd_combine_<cutlass::half_t, 256>(params, stream);
        }
    }
}

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        // batch_size x seqlen_k x num_heads_k x head_size or num_pages x page_size x num_heads_k x head_size if there's a page_table.
        const at::Tensor &k,
        const at::Tensor &v,
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        const float softmax_scale,
        bool is_causal,
        c10::optional<at::Tensor> &q_descale_,  // 1
        c10::optional<at::Tensor> &k_descale_,  // 1
        c10::optional<at::Tensor> &v_descale_,  // 1
        int window_size_left,
        int window_size_right,
        int sink_token_length,
        const float softcap,
        int num_splits,
        c10::optional<bool> pack_gqa_
        ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90, "FlashAttention only supports Hopper GPUs or newer.");

    auto q_type = q.scalar_type();
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 || q_type == at::ScalarType::Float8_e4m3fn,
                "FlashAttention only support fp16, bf16, and fp8_e4m3 data type");
    TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1 || v.stride(-3) == 1, "Input tensor V must have contiguous last dimension or contiguous seqlen dimension");
    if (v.stride(-1) != 1) {
        TORCH_CHECK(q_type == at::ScalarType::Float8_e4m3fn, "Only fp8_e4m3 data type supports input tensor V having contiguous seqlen dimension")
    }

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // TODO: check this
    if (window_size_left >= seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= seqlen_q - 1) { window_size_right = -1; }
    if (is_causal) {
        window_size_left = -1;
        window_size_right = 0;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

    int const alignment = q_type == torch::kFloat8_e4m3fn ? 16 : 8;
    at::Tensor q_padded, k_padded, v_padded;
    auto pad = [](at::Tensor x, int alignment) {
        return x.size(-1) % alignment == 0 ? x : torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, alignment - x.size(-1) % alignment}));
    };
    q_padded = pad(q, alignment);
    k_padded = pad(k, alignment);
    v_padded = pad(v, alignment);

    if (v_padded.stride(-1) != 1) {
        TORCH_CHECK(v_padded.stride(-1) % 16 == 0 && v_padded.stride(-2) % 16 == 0 && v_padded.stride(-4) % 16 == 0,
                    "If input tensor V has contiguous seqlen dimension, the others dimension must have stride divisible by 16");
    }

    auto opts = q.options();
    auto out_type = q_type == at::ScalarType::Float8_e4m3fn ? at::ScalarType::BFloat16 : q_type;
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == out_type, "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, output must have dtype BF16");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        if (head_size_og % alignment != 0) { out = torch::empty_like(q_padded, opts.dtype(out_type)); }
    } else {
        out = torch::empty_like(q_padded, opts.dtype(out_type));
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, alignment);
    const int head_size_rounded = head_size <= 64 ? 64 : (head_size <= 128 ? round_multiple(head_size, 32) : round_multiple(head_size, 64));
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_q_=*/nullptr,
                     /*seqused_k=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap);
    params.sink_token_length = sink_token_length;

    params.num_splits = num_splits;
    at::Tensor out_accum, softmax_lse_accum;
    auto outaccum_type = at::ScalarType::Float;
    if (num_splits > 1) {
        TORCH_CHECK(params.num_splits <= 256, "num_splits > 256 not supported");
        out_accum = torch::empty({num_splits, batch_size, num_heads, seqlen_q, head_size}, opts.dtype(outaccum_type));
        softmax_lse_accum = torch::empty({num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
        params.is_fp32 = false;
        params.oaccum_ptr = out_accum.data_ptr();
        params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        params.oaccum_split_stride = out_accum.stride(0);
        params.oaccum_row_stride = out_accum.stride(3);
        params.oaccum_head_stride = out_accum.stride(2);
        params.oaccum_batch_stride = out_accum.stride(1);
        params.lseaccum_split_stride = softmax_lse_accum.stride(0);
        params.lseaccum_head_stride = softmax_lse_accum.stride(2);
        params.lseaccum_batch_stride = softmax_lse_accum.stride(1);
    }

    params.pack_gqa = pack_gqa_.has_value() ? int (pack_gqa_.value()) : -1;

    auto tile_count_semaphore = (params.is_causal || params.is_local) ? torch::zeros({1}, opts.dtype(torch::kInt32)) : torch::empty({1}, opts.dtype(torch::kInt32));
    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

    if (q_type == at::ScalarType::Float8_e4m3fn) {
        if (q_descale_.has_value()) {
            auto q_descale = q_descale_.value();
            CHECK_DEVICE(q_descale);
            CHECK_SHAPE(q_descale, 1);
            params.q_descale_ptr = q_descale.data_ptr<float>();
        } else {
            params.q_descale_ptr = nullptr;
        }
        if (k_descale_.has_value()) {
            auto k_descale = k_descale_.value();
            CHECK_DEVICE(k_descale);
            CHECK_SHAPE(k_descale, 1);
            params.k_descale_ptr = k_descale.data_ptr<float>();
        } else {
            params.k_descale_ptr = nullptr;
        }
        if (v_descale_.has_value()) {
            auto v_descale = v_descale_.value();
            CHECK_DEVICE(v_descale);
            CHECK_SHAPE(v_descale, 1);
            params.v_descale_ptr = v_descale.data_ptr<float>();
        } else {
            params.v_descale_ptr = nullptr;
        }
    }

    if (seqlen_k > 0 && batch_size > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
        if (num_splits > 1) {
            params.is_bf16 = true;  // Since we want output in BF16. Otherwise fwd_combine will output to FP16
            run_mha_fwd_combine(params, stream);
        }
    } else if (batch_size > 0) {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og % alignment != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse};
    // return {out, q_padded, k_padded, v_padded, out_accum, softmax_lse_accum};
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
               c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
               c10::optional<at::Tensor> &seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
               int const max_seqlen_q,
               int const max_seqlen_k,
               const float softmax_scale,
               bool is_causal,
               c10::optional<at::Tensor> &q_descale_,  // 1
               c10::optional<at::Tensor> &k_descale_,  // 1
               c10::optional<at::Tensor> &v_descale_,  // 1
               int window_size_left,
               int window_size_right,
               const float softcap,
               int num_splits,
               c10::optional<bool> pack_gqa_
               ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90, "FlashAttention only supports Hopper GPUs or newer.");

    auto q_type = q.scalar_type();
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 || q_type == at::ScalarType::Float8_e4m3fn,
                "FlashAttention only support fp16, bf16, and fp8_e4m3 data type");
    TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(cu_seqlens_q.stride(-1) == 1, "cu_seqlens_q must have contiguous last dimension");
    TORCH_CHECK(cu_seqlens_k.stride(-1) == 1, "cu_seqlens_q must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int num_heads_k = k.size(1);
    const int total_q = q.sizes()[0];
    const int total_k = k.sizes()[0];
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= max_seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_q - 1) { window_size_right = -1; }
    if (is_causal) {
        window_size_left = -1;
        window_size_right = 0;
    }

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    if (seqused_q_.has_value()){
        auto seqused_q = seqused_q_.value();
        TORCH_CHECK(seqused_q.dtype() == torch::kInt32, "seqused_q must have dtype int32");
        TORCH_CHECK(seqused_q.is_cuda(), "seqused_q must be on CUDA device");
        TORCH_CHECK(seqused_q.is_contiguous(), "seqused_q must be contiguous");
        CHECK_SHAPE(seqused_q, batch_size);
    }
    if (seqused_k_.has_value()){
        auto seqused_k = seqused_k_.value();
        TORCH_CHECK(seqused_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k, batch_size);
    }

    int const alignment = q_type == torch::kFloat8_e4m3fn ? 16 : 8;

    at::Tensor q_padded, k_padded, v_padded;
    auto pad = [](at::Tensor x, int alignment) {
        return x.size(-1) % alignment == 0 ? x : torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, alignment - x.size(-1) % alignment}));
    };
    q_padded = pad(q, alignment);
    k_padded = pad(k, alignment);
    v_padded = pad(v, alignment);

    auto opts = q.options();
    auto out_type = q_type == at::ScalarType::Float8_e4m3fn ? at::ScalarType::BFloat16 : q_type;
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == out_type, "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, output must have dtype BF16");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, total_q, num_heads, head_size_og);
        if (head_size_og % alignment != 0) { out = torch::empty_like(q_padded, opts.dtype(out_type)); }
    } else {
        out = torch::empty_like(q_padded, opts.dtype(out_type));
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, alignment);
    const int head_size_rounded = head_size <= 64 ? 64 : (head_size <= 128 ? round_multiple(head_size, 32) : round_multiple(head_size, 64));
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     seqused_q_.has_value() ? seqused_q_.value().data_ptr() : nullptr,
                     seqused_k_.has_value() ? seqused_k_.value().data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap);
    params.total_q = total_q;
    params.total_k = total_k;

    params.num_splits = num_splits;
    at::Tensor out_accum, softmax_lse_accum;
    auto outaccum_type = at::ScalarType::Float;
    if (num_splits > 1) {
        TORCH_CHECK(params.num_splits <= 256, "num_splits > 256 not supported");
        out_accum = torch::empty({num_splits, num_heads, total_q, head_size}, opts.dtype(outaccum_type));
        softmax_lse_accum = torch::empty({num_splits, num_heads, total_q}, opts.dtype(at::kFloat));
        params.is_fp32 = false;
        params.oaccum_ptr = out_accum.data_ptr();
        params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        params.oaccum_split_stride = out_accum.stride(0);
        params.oaccum_row_stride = out_accum.stride(2);
        params.oaccum_head_stride = out_accum.stride(1);
        params.oaccum_batch_stride = 0;
        params.lseaccum_split_stride = softmax_lse_accum.stride(0);
        params.lseaccum_head_stride = softmax_lse_accum.stride(1);
        params.lseaccum_batch_stride = 0;
    }

    // If negative, we use a heuristic to decide
    params.pack_gqa = pack_gqa_.has_value() ? int(pack_gqa_.value()) : -1;

    auto tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));
    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

    if (q_type == at::ScalarType::Float8_e4m3fn) {
        if (q_descale_.has_value()) {
            auto q_descale = q_descale_.value();
            CHECK_DEVICE(q_descale);
            CHECK_SHAPE(q_descale, 1);
            params.q_descale_ptr = q_descale.data_ptr<float>();
        } else {
            params.q_descale_ptr = nullptr;
        }
        if (k_descale_.has_value()) {
            auto k_descale = k_descale_.value();
            CHECK_DEVICE(k_descale);
            CHECK_SHAPE(k_descale, 1);
            params.k_descale_ptr = k_descale.data_ptr<float>();
        } else {
            params.k_descale_ptr = nullptr;
        }
        if (v_descale_.has_value()) {
            auto v_descale = v_descale_.value();
            CHECK_DEVICE(v_descale);
            CHECK_SHAPE(v_descale, 1);
            params.v_descale_ptr = v_descale.data_ptr<float>();
        } else {
            params.v_descale_ptr = nullptr;
        }
    }

    if (max_seqlen_k > 0 && batch_size > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
        if (num_splits > 1) {
            params.is_bf16 = true;  // Since we want output in BF16. Otherwise fwd_combine will output to FP16
            // Unless there's seqused_q, for the purpose of attn_combine, we can just treat it as batch=1
            // and seqlen = total_q, and don't need to dispatch to Varlen there.
            if (!seqused_q_.has_value()) {
                params.b = 1;
                params.seqlen_q = total_q;
            }
            run_mha_fwd_combine(params, stream);
        }
    } else if (batch_size > 0) {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse};
    // return {out, q_padded, k_padded, v_padded, out_accum, softmax_lse_accum};
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    // FP16_SWITCH(!params.is_bf16, [&] {
    //     HEADDIM_SWITCH(params.d, [&] {
    //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
    //     });
    // });
    if (!params.is_bf16) {
        if (params.d <= 64) {
            run_mha_bwd_<cutlass::half_t, 64>(params, stream);
        } else if (params.d <= 96) {
            run_mha_bwd_<cutlass::half_t, 96>(params, stream);
        } else if (params.d <= 128) {
            run_mha_bwd_<cutlass::half_t, 128>(params, stream);
        } else if (params.d <= 192) {
            run_mha_bwd_<cutlass::half_t, 192>(params, stream);
        } else {
            run_mha_bwd_<cutlass::half_t, 256>(params, stream);
        }
    } else {
        if (params.d <= 64) {
            run_mha_bwd_<cutlass::bfloat16_t, 64>(params, stream);
        } else if (params.d <= 96) {
            run_mha_bwd_<cutlass::bfloat16_t, 96>(params, stream);
        } else if (params.d <= 128) {
            run_mha_bwd_<cutlass::bfloat16_t, 128>(params, stream);
        } else if (params.d <= 192) {
            run_mha_bwd_<cutlass::bfloat16_t, 192>(params, stream);
        } else {
            run_mha_bwd_<cutlass::bfloat16_t, 256>(params, stream);
        }
    }
}

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        int sink_token_length,
        const float softcap,
        const bool deterministic) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm9x = dprops->major == 9 && dprops->minor >= 0;
    TORCH_CHECK(is_sm9x, "FlashAttentionHopper only supports Hopper GPUs or newer.");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_type = q.dtype();
    TORCH_CHECK(q_type == torch::kFloat16 || q_type == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_type, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_type, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_type, "query and dout must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = dout.size(3);
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 64 ? 64 : (head_size <= 128 ? round_multiple(head_size, 32) : round_multiple(head_size, 64));
    // This needs to match the kernel configs
    const int kBlockM = head_size_rounded <= 64 ? (softcap == 0.0 ? 128 : 96) : 64;
    const int kBlockN = head_size_rounded <= 128 ? 128 : (head_size_rounded <= 192 ? 96 : 80);
    const int seqlen_q_rounded = round_multiple(seqlen_q, kBlockM);
    const int seqlen_k_rounded = round_multiple(seqlen_k, kBlockN);

    TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

    if (window_size_left >= seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= seqlen_q - 1) { window_size_right = -1; }
    if (is_causal) {
        window_size_left = -1;
        window_size_right = 0;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_type, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_type, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_type, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    // Need softmax_d to have seqlen_q_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    auto softmax_lse_log2 = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    // dq_accum = torch::empty({batch_size, num_heads, seqlen_q_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    dq_accum = torch::zeros({batch_size, num_heads, seqlen_q_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_accum = torch::zeros({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
        dv_accum = torch::zeros({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout_padded, dq, dk, dv,
                     nullptr /*cu_seqlens_q*/,
                     nullptr /*cu_seqlens_k*/,
                     nullptr /*seqused_q_*/,
                     nullptr /*seqused_k*/,
                     dq_accum.data_ptr(),
                     num_heads_k != num_heads ? dk_accum.data_ptr() : nullptr,
                     num_heads_k != num_heads ? dv_accum.data_ptr() : nullptr,
                     // nullptr,
                     // nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     deterministic);
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();
    params.sink_token_length = sink_token_length;

    // Will be zero'ed out in the backward preprocess kernel
    at::Tensor dq_semaphore = torch::empty({(seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads}, opts.dtype(torch::kInt32));
    params.dq_semaphore = dq_semaphore.data_ptr<int>();
    // printf("dq_semaphore: %p, [%d, %d, %d]\n", params.dq_semaphore, (seqlen_q + 64 - 1) / 64, batch_size, num_heads);
    if (num_heads_k != num_heads) {
        at::Tensor dk_semaphore = torch::zeros({(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, opts.dtype(torch::kInt32));
        at::Tensor dv_semaphore = torch::zeros({(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, opts.dtype(torch::kInt32));
        params.dk_semaphore = dk_semaphore.data_ptr<int>();
        params.dv_semaphore = dv_semaphore.data_ptr<int>();
    }

    if (seqlen_q > 0) {
        run_mha_bwd(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk.zero_();
        dv.zero_();
        softmax_d.zero_();
    }

    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    }

    return { dq, dk, dv, softmax_d, dq_accum, dk_accum, dv_accum};
}

std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
               const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &softmax_lse,     // b x h x seqlen_q
               c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
               c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
               c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
               c10::optional<at::Tensor> &seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float softmax_scale,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool deterministic) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm9x = dprops->major == 9 && dprops->minor >= 0;
    TORCH_CHECK(is_sm9x, "FlashAttentionHopper only supports Hopper GPUs or newer.");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_type = q.dtype();
    TORCH_CHECK(q_type == torch::kFloat16 || q_type == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_type, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_type, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_type, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size_og = dout.size(2);
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 64 ? 64 : (head_size <= 128 ? round_multiple(head_size, 32) : round_multiple(head_size, 64));
    // This needs to match the kernel configs
    const int kBlockM = head_size_rounded <= 64 ? (softcap == 0.0 ? 128 : 96) : 64;
    const int kBlockN = head_size_rounded <= 128 ? 128 : (head_size_rounded <= 192 ? 96 : 80);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, kBlockM);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, kBlockN);
    int const total_q_padded_rounded = round_multiple(total_q + batch_size * kBlockM, kBlockM);
    int const total_k_padded_rounded = round_multiple(total_k + batch_size * kBlockN, kBlockN);

    TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

    if (window_size_left >= max_seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_q - 1) { window_size_right = -1; }
    if (is_causal) {
        window_size_left = -1;
        window_size_right = 0;
    }

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    if (seqused_q_.has_value()){
        auto seqused_q = seqused_q_.value();
        TORCH_CHECK(seqused_q.dtype() == torch::kInt32, "seqused_q must have dtype int32");
        TORCH_CHECK(seqused_q.is_cuda(), "seqused_q must be on CUDA device");
        TORCH_CHECK(seqused_q.is_contiguous(), "seqused_q must be contiguous");
        CHECK_SHAPE(seqused_q, batch_size);
    }
    if (seqused_k_.has_value()){
        auto seqused_k = seqused_k_.value();
        TORCH_CHECK(seqused_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k, batch_size);
    }

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_type, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_type, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_type, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    // Need softmax_d to have total_q_padded_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
    auto softmax_d = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
    auto softmax_lse_log2 = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    dq_accum = torch::empty({num_heads, total_q_padded_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_accum = torch::zeros({num_heads_k, total_k_padded_rounded, head_size_rounded}, opts.dtype(at::kFloat));
        dv_accum = torch::zeros({num_heads_k, total_k_padded_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout_padded, dq, dk, dv,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     seqused_q_.has_value() ? seqused_q_.value().data_ptr() : nullptr,
                     seqused_k_.has_value() ? seqused_k_.value().data_ptr() : nullptr,
                     dq_accum.data_ptr(),
                     num_heads_k != num_heads ? dk_accum.data_ptr() : nullptr,
                     num_heads_k != num_heads ? dv_accum.data_ptr() : nullptr,
                     // nullptr,
                     // nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     deterministic);
    params.total_q = total_q;
    params.total_k = total_k;
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();

    // Will be zero'ed out in the backward preprocess kernel
    at::Tensor dq_semaphore = torch::empty({(max_seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads}, opts.dtype(torch::kInt32));
    params.dq_semaphore = dq_semaphore.data_ptr<int>();
    if (num_heads_k != num_heads) {
        // TODO: do we need to zero them out?
        at::Tensor dk_semaphore = torch::empty({(max_seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, opts.dtype(torch::kInt32));
        at::Tensor dv_semaphore = torch::empty({(max_seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, opts.dtype(torch::kInt32));
        params.dk_semaphore = dk_semaphore.data_ptr<int>();
        params.dv_semaphore = dv_semaphore.data_ptr<int>();
    }

    if (max_seqlen_q > 0) {
        run_mha_bwd(params, stream);
    } else {
        // If max_seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk.zero_();
        dv.zero_();
        softmax_d.zero_();
    }

    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    }

    return { dq, dk, dv, softmax_d, dq_accum, softmax_lse_log2 };
}

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size or total_q x num_heads x head_size
                // batch_size_k x seqlen_k x num_heads_k x head_size or num_pages x page_size x num_heads_k x head_size if there's a page_table.
                const at::Tensor &kcache,
                const at::Tensor &vcache,
                c10::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                // batch_size x seqlen_q x num_heads x head_size or total_q x num_heads x head_size
                c10::optional<at::Tensor> &out_,
                c10::optional<const at::Tensor> &seqused_k_, // batch_size
                c10::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                c10::optional<const at::Tensor> &leftpad_k_, // batch_size
                c10::optional<const at::Tensor> &page_table_, // batch_size_k x max_num_pages_per_seq
                c10::optional<const at::Tensor> &cu_seqlens_q_,  // b+1
                c10::optional<int> max_seqlen_q_,
                const float softmax_scale,
                bool is_causal,
                c10::optional<at::Tensor> &q_descale_,  // 1
                c10::optional<at::Tensor> &k_descale_,  // 1
                c10::optional<at::Tensor> &v_descale_,  // 1
                int window_size_left,
                int window_size_right,
                int sink_token_length,
                const float softcap,
                int num_splits,
                c10::optional<bool> pack_gqa_
                ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90, "FlashAttention only supports Hopper GPUs or newer.");

    auto q_type = q.scalar_type();
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 || q_type == at::ScalarType::Float8_e4m3fn,
                "FlashAttention only support fp16, bf16, and fp8_e4m3 data type");
    TORCH_CHECK(kcache.scalar_type() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(vcache.scalar_type() == q_type, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor page_table;
    const bool paged_KV = page_table_.has_value();
    if (paged_KV) {
        page_table = page_table_.value();
        CHECK_DEVICE(page_table);
        TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype torch.int32");
        TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");
    }

    at::Tensor cu_seqlens_q;
    bool const is_varlen_q = cu_seqlens_q_.has_value();
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_DEVICE(cu_seqlens_q);
        TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");
        TORCH_CHECK(cu_seqlens_q.stride(-1) == 1, "cu_seqlens_q must have contiguous last dimension");
        TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }

    const auto sizes = q.sizes();

    const int batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q.size(0) - 1;
    int seqlen_q = !is_varlen_q ? sizes[1] : max_seqlen_q_.value();
    int total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
    int num_heads = q.size(-2);
    const int head_size_og = q.size(-1);
    const int num_heads_k = kcache.size(2);
    const int batch_size_k = !paged_KV ? kcache.size(0) : page_table.size(0);
    if (!cache_batch_idx_.has_value()) {
        TORCH_CHECK(batch_size == batch_size_k, "batch_size must be equal to batch_size_k");
    }
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    const int max_num_pages_per_seq = !paged_KV ? 0 : page_table.size(1);
    const int num_pages = !paged_KV ? 0 : kcache.size(0);
    const int page_size = !paged_KV ? 1 : kcache.size(1);

    const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_pages_per_seq * page_size;

    // TODO: check this
    if (window_size_left >= seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= seqlen_q - 1) { window_size_right = -1; }
    if (is_causal) {
        window_size_left = -1;
        window_size_right = 0;
    }

    if (!is_varlen_q) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    } else {
        CHECK_SHAPE(q, total_q, num_heads, head_size_og);
        CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    }
    if (!paged_KV) {
        CHECK_SHAPE(kcache, batch_size_k, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, batch_size_k, seqlen_k, num_heads_k, head_size_og);
    } else {
        CHECK_SHAPE(kcache, num_pages, page_size, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, num_pages, page_size, num_heads_k, head_size_og);
        CHECK_SHAPE(page_table, batch_size_k, max_num_pages_per_seq);
    }

    if (seqused_k_.has_value()) {
        auto seqused_k = seqused_k_.value();
        TORCH_CHECK(seqused_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        CHECK_DEVICE(seqused_k);
        CHECK_CONTIGUOUS(seqused_k);
        CHECK_SHAPE(seqused_k, batch_size);
    }

    int const alignment = q_type == torch::kFloat8_e4m3fn ? 16 : 8;
    at::Tensor q_padded, k_padded, v_padded;
    auto pad = [](at::Tensor x, int alignment) {
        return x.size(-1) % alignment == 0 ? x : torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, alignment - x.size(-1) % alignment}));
    };
    q_padded = pad(q, alignment);
    k_padded = pad(kcache, alignment);
    v_padded = pad(vcache, alignment);

    auto opts = q.options();
    auto out_type = q_type == at::ScalarType::Float8_e4m3fn ? at::ScalarType::BFloat16 : q_type;
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == out_type, "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, output must have dtype BF16");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if (!is_varlen_q) {
            CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        } else {
            CHECK_SHAPE(out, total_q, num_heads, head_size_og);
        }
        if (head_size_og % alignment != 0) { out = torch::empty_like(q_padded, opts.dtype(out_type)); }
    } else {
        out = torch::empty_like(q_padded, opts.dtype(out_type));
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, alignment);
    const int head_size_rounded = head_size <= 64 ? 64 : (head_size <= 128 ? round_multiple(head_size, 32) : round_multiple(head_size, 64));
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     /*cu_seqlens_q_d=*/!is_varlen_q ? nullptr : cu_seqlens_q.data_ptr(),
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_q_=*/nullptr,
                     /*seqused_k=*/seqused_k_.has_value() ? seqused_k_.value().data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap);
    params.total_q = total_q;
    params.sink_token_length = sink_token_length;
    params.b_k = batch_size_k;

    params.num_splits = num_splits;
    TORCH_CHECK(num_splits >= 1, "num_splits must be at least 1, there's no heuristic to automatically pick num_splits yet");
    at::Tensor out_accum, softmax_lse_accum;
    auto outaccum_type = at::ScalarType::Float;
    if (num_splits > 1) {
        TORCH_CHECK(params.num_splits <= 256, "num_splits > 256 not supported");
        if (!is_varlen_q) {
            out_accum = torch::empty({num_splits, batch_size, num_heads, seqlen_q, head_size}, opts.dtype(outaccum_type));
            softmax_lse_accum = torch::empty({num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
            params.oaccum_batch_stride = out_accum.stride(1);
            params.lseaccum_batch_stride = softmax_lse_accum.stride(1);
        } else {
            out_accum = torch::empty({num_splits, num_heads, total_q, head_size}, opts.dtype(outaccum_type));
            softmax_lse_accum = torch::empty({num_splits, num_heads, total_q}, opts.dtype(at::kFloat));
        }
        params.is_fp32 = false;
        params.oaccum_ptr = out_accum.data_ptr();
        params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        params.oaccum_split_stride = out_accum.stride(0);
        params.oaccum_row_stride = out_accum.stride(-2);
        params.oaccum_head_stride = out_accum.stride(-3);
        params.lseaccum_split_stride = softmax_lse_accum.stride(0);
        params.lseaccum_head_stride = softmax_lse_accum.stride(-2);
    }

    if (paged_KV) {
        params.page_table = page_table.data_ptr<int>();
        params.page_table_batch_stride = page_table.stride(0);
    }
    params.page_size = page_size;
    params.num_pages = num_pages;

    if (k_.has_value()) {
        at::Tensor k, v, k_padded, v_padded;
        TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        TORCH_CHECK(seqused_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
        TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        k = k_.value();
        v = v_.value();
        TORCH_CHECK(k.dtype() == q_type, "Key must have the same dtype as query");
        TORCH_CHECK(v.dtype() == q_type, "Value must have the same dtype as query");
        CHECK_DEVICE(k); CHECK_DEVICE(v);
        TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        int seqlen_knew = k.size(1);
        CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
        if (head_size_og % 8 != 0) {
            k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        } else {
            k_padded = k;
            v_padded = v;
        }
        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = k_padded.data_ptr();
        params.vnew_ptr = v_padded.data_ptr();
        // All stride are in elements, not bytes.
        params.knew_batch_stride = k_padded.stride(0);
        params.vnew_batch_stride = v_padded.stride(0);
        params.knew_row_stride = k_padded.stride(-3);
        params.vnew_row_stride = v_padded.stride(-3);
        params.knew_head_stride = k_padded.stride(-2);
        params.vnew_head_stride = v_padded.stride(-2);
    }

    if (leftpad_k_.has_value()) {
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    if (cache_batch_idx_.has_value()) {
        auto cache_batch_idx = cache_batch_idx_.value();
        CHECK_DEVICE(cache_batch_idx);
        CHECK_CONTIGUOUS(cache_batch_idx);
        TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
        params.kv_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
    }

    params.pack_gqa = pack_gqa_.has_value() ? int (pack_gqa_.value()) : -1;

    at::Tensor tile_count_semaphore;
    // We don't use the persistent scheduler if Split or PagedKV
    if ((params.is_causal || params.is_local || seqused_k_.has_value() || leftpad_k_.has_value()) && params.num_splits == 1 && !paged_KV) {
        tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));
        params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
    } else {
        params.tile_count_semaphore = nullptr;
    }

    if (q_type == at::ScalarType::Float8_e4m3fn) {
        if (q_descale_.has_value()) {
            auto q_descale = q_descale_.value();
            CHECK_DEVICE(q_descale);
            CHECK_SHAPE(q_descale, 1);
            params.q_descale_ptr = q_descale.data_ptr<float>();
        } else {
            params.q_descale_ptr = nullptr;
        }
        if (k_descale_.has_value()) {
            auto k_descale = k_descale_.value();
            CHECK_DEVICE(k_descale);
            CHECK_SHAPE(k_descale, 1);
            params.k_descale_ptr = k_descale.data_ptr<float>();
        } else {
            params.k_descale_ptr = nullptr;
        }
        if (v_descale_.has_value()) {
            auto v_descale = v_descale_.value();
            CHECK_DEVICE(v_descale);
            CHECK_SHAPE(v_descale, 1);
            params.v_descale_ptr = v_descale.data_ptr<float>();
        } else {
            params.v_descale_ptr = nullptr;
        }
    }

    if (seqlen_q > 0 && total_q > 0 && seqlen_k > 0 && batch_size > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
        if (num_splits > 1) {
            params.is_bf16 = true;  // Since we want output in BF16. Otherwise fwd_combine will output to FP16
            // Unless there's seqused_q, for the purpose of attn_combine, we can just treat it as batch=1
            // and seqlen = total_q, and don't need to dispatch to Varlen there.
            // if (is_varlen_q && !seqused_q_.has_value()) {
            if (is_varlen_q) {
                params.b = 1;
                params.seqlen_q = total_q;
            }
            run_mha_fwd_combine(params, stream);
        }
    } else if (seqlen_q > 0 && total_q > 0 && batch_size > 0) {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og % alignment != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    // return {out, softmax_lse};
    return {out, softmax_lse, out_accum, softmax_lse_accum};
}

std::vector<at::Tensor>
mha_combine(const at::Tensor &out_partial,         // num_splits x batch_size x seqlen x num_heads x head_size
            const at::Tensor &lse_partial,         // num_splits x batch_size x seqlen x num_heads
            std::optional<at::Tensor> out_,        // batch_size x seqlen x num_heads x head_size
            std::optional<at::ScalarType> out_dtype_
            ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm80 = dprops->major >= 8;
    TORCH_CHECK(is_sm80, "Attention combine function only supports Ampere GPUs or newer.");

    auto out_partial_type = out_partial.scalar_type();
    TORCH_CHECK(out_partial_type == at::ScalarType::Float, "Attention combine function only support fp32 data type");
    TORCH_CHECK(lse_partial.scalar_type() == at::ScalarType::Float, "Attention combine function only support fp32 data type");

    CHECK_DEVICE(out_partial); CHECK_DEVICE(lse_partial);

    TORCH_CHECK(out_partial.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(lse_partial.stride(-2) == 1, "LSE tensor must be contiguous in the seqlen dimension");

    const auto sizes = out_partial.sizes();

    const int num_splits = sizes[0];
    const int batch_size = sizes[1];
    const int seqlen = sizes[2];
    const int num_heads = sizes[3];
    const int head_size_og = sizes[4];
    TORCH_CHECK(head_size_og <= 256, "FlashAttention combine only supports head dimension at most 256");
    TORCH_CHECK(num_splits <= 256, "FlashAttention combine only supports num_splits at most 256");

    CHECK_SHAPE(out_partial, num_splits, batch_size, seqlen, num_heads, head_size_og);
    CHECK_SHAPE(lse_partial, num_splits, batch_size, seqlen, num_heads);

    int const alignment = 4;
    at::Tensor out_partial_padded;
    auto pad = [](at::Tensor x, int alignment) {
        return x.size(-1) % alignment == 0 ? x : torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, alignment - x.size(-1) % alignment}));
    };
    out_partial_padded = pad(out_partial, alignment);

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, alignment);

    auto opts = out_partial.options();
    at::ScalarType out_type = out_dtype_.value_or(out_partial.scalar_type());
    TORCH_CHECK(out_type == at::ScalarType::Float || out_type == at::ScalarType::BFloat16 || out_type == at::ScalarType::Half, "Output type must be FP32, FP16 or BF16");
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == out_type);
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen, num_heads, head_size_og);
        if (head_size_og % alignment != 0) {
            out = torch::empty({batch_size, seqlen, num_heads, head_size}, opts.dtype(out_type));
        }
    } else {
        out = torch::empty({batch_size, seqlen, num_heads, head_size}, opts.dtype(out_type));
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)out_partial.get_device()};

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen}, opts.dtype(at::kFloat)).transpose(1, 2);

    Flash_fwd_params params {};  // Need to reset the params to set everything to zero
    params.is_fp32 = out_type == at::ScalarType::Float;
    params.is_bf16 = out_type == at::ScalarType::BFloat16;
    params.oaccum_ptr = out_partial_padded.data_ptr();
    params.softmax_lseaccum_ptr = lse_partial.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    params.b = batch_size;
    params.h = num_heads;
    params.seqlen_q = seqlen;
    params.d = head_size;
    params.num_splits = num_splits;
    params.oaccum_split_stride = out_partial_padded.stride(0);
    params.oaccum_row_stride = out_partial_padded.stride(2);
    params.oaccum_head_stride = out_partial_padded.stride(3);
    params.oaccum_batch_stride = out_partial_padded.stride(1);
    params.lseaccum_split_stride = lse_partial.stride(0);
    params.lseaccum_head_stride = lse_partial.stride(3);
    params.lseaccum_batch_stride = lse_partial.stride(1);
    params.o_row_stride = out.stride(1);
    params.o_head_stride = out.stride(2);
    params.o_batch_stride = out.stride(0);

    if (seqlen > 0 && batch_size > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd_combine(params, stream);
    }

    at::Tensor out_padded = out;
    if (head_size_og % alignment != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        // if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, softmax_lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("fwd_varlen", &mha_varlen_fwd, "Varlen forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("bwd_varlen", &mha_varlen_bwd, "Varlen backward pass");
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
    m.def("fwd_combine", &mha_combine, "Combine partial attention outputs");
}
