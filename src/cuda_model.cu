#include "cuda_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "cuda_util.cuh"
#include "quant.h"

namespace {

constexpr int kThreads = 256;  // every kernel here; a multiple of the warp size

int ceil_div(std::size_t x, int d) { return static_cast<int>((x + d - 1) / d); }

void check_launch() { cuda_check(cudaGetLastError(), "launch"); }

// The kernels below read the call's first position from feed[0] and its
// token ids from feed[1..n], in device memory, so a captured graph is valid
// at any position.

// x[i] = table[feed[1 + i]], one block per row.
__global__ void embed(int dim, const int* feed, const float* table, float* x) {
    const float* src = table + static_cast<std::size_t>(feed[1 + blockIdx.x]) * dim;
    float* dst = x + static_cast<std::size_t>(blockIdx.x) * dim;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) dst[j] = src[j];
}

// For row blockIdx.y, fed at position feed[0] + blockIdx.y: rotates each
// (even, odd) pair of q and k by the position's angle, then writes the
// rotated k and the v slice into the caches. A thread takes one q pair, one
// k pair or one v element.
__global__ void rope_and_cache(int qkv_cols, int dim, int kv_dim, int head_size,
                               const int* feed, const float* cos_t, const float* sin_t,
                               float* qkv, float* key_cache, float* value_cache) {
    const int i = blockIdx.y, pos = feed[0] + i, half = head_size / 2;
    float* q = qkv + static_cast<std::size_t>(i) * qkv_cols;
    float* k = q + dim;
    const float* v = k + kv_dim;
    const int nq = dim / 2, nk = kv_dim / 2;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nq + nk) {
        const bool is_k = j >= nq;
        const int pair = is_k ? j - nq : j;
        float* p = (is_k ? k : q) + 2 * pair;
        const float c = cos_t[pos * half + pair % half], s = sin_t[pos * half + pair % half];
        const float a = p[0], b = p[1];
        p[0] = a * c - b * s;
        p[1] = a * s + b * c;
        if (is_k) {
            float* dst = key_cache + static_cast<std::size_t>(pos) * kv_dim + 2 * pair;
            dst[0] = p[0];
            dst[1] = p[1];
        }
    } else if (j < nq + nk + kv_dim) {
        const int t = j - nq - nk;
        value_cache[static_cast<std::size_t>(pos) * kv_dim + t] = v[t];
    }
}

// Causal attention, one block per (query head, row): row i, at position
// feed[0] + i, attends to cached positions [0, feed[0] + i]. The query and
// the scores live in shared memory (head_size + seq_len floats). Threads
// split the positions for the scores and the head's dimensions for the
// weighted sum of values, which reads each value row coalesced. (Splitting
// the weighted sum across all 256 threads in head_size-wide groups, with a
// final reduction, was slower: 11.9 vs 9.8 us per call.)
__global__ void attention(int qkv_cols, int dim, int kv_dim, int head_size, int group,
                          const int* feed, const float* qkv, const float* key_cache,
                          const float* value_cache, float* out) {
    extern __shared__ float smem[];
    float* q = smem;
    float* scores = smem + head_size;
    const int h = blockIdx.x, i = blockIdx.y, n_ctx = feed[0] + i + 1;
    const int kv_off = (h / group) * head_size;

    const float* qg = qkv + static_cast<std::size_t>(i) * qkv_cols + h * head_size;
    for (int d = threadIdx.x; d < head_size; d += blockDim.x) q[d] = qg[d];
    __syncthreads();

    const float scale = 1.0f / sqrtf(float(head_size));
    float mx = -INFINITY;
    for (int t = threadIdx.x; t < n_ctx; t += blockDim.x) {
        const float* k = key_cache + static_cast<std::size_t>(t) * kv_dim + kv_off;
        float dot = 0.0f;
        for (int d = 0; d < head_size; ++d) dot += q[d] * k[d];
        scores[t] = dot * scale;
        mx = fmaxf(mx, scores[t]);
    }
    mx = block_reduce(mx, MaxOp{}, -INFINITY);
    float sum = 0.0f;
    for (int t = threadIdx.x; t < n_ctx; t += blockDim.x) {
        scores[t] = expf(scores[t] - mx);
        sum += scores[t];
    }
    // block_reduce's barriers also make every score visible below.
    const float inv = 1.0f / block_reduce(sum, SumOp{}, 0.0f);

    float* oh = out + static_cast<std::size_t>(i) * dim + h * head_size;
    for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
        const float* v = value_cache + kv_off + d;
        float acc = 0.0f;
        for (int t = 0; t < n_ctx; ++t) acc += scores[t] * v[static_cast<std::size_t>(t) * kv_dim];
        oh[d] = acc * inv;
    }
}

// SwiGLU: hb[r][c] = silu(gu[r][c]) * gu[r][hidden + c].
__global__ void swiglu(int rows, int hidden, const float* gu, float* hb) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * hidden) return;
    const int r = idx / hidden, c = idx % hidden;
    const float g = gu[r * 2 * hidden + c];
    hb[idx] = g / (1.0f + expf(-g)) * gu[r * 2 * hidden + hidden + c];
}

DeviceTensor upload(const std::vector<float>& v) {
    Tensor t(1, v.size());
    t.data = v;
    return DeviceTensor(t);
}

MatmulFusion rms(const DeviceTensor& weight) {
    MatmulFusion f;
    f.rms_weight = weight.data();
    return f;
}

MatmulFusion residual() {
    MatmulFusion f;
    f.accumulate = true;
    return f;
}

// Shared memory for attention: the query and the scores.
std::size_t attention_smem(const Config& c) {
    return (c.head_size() + c.seq_len) * sizeof(float);
}

} // namespace

CudaTransformer::CudaTransformer(const Checkpoint& ckpt, WeightType type)
    : cfg_(ckpt.config), quantized_(type == WeightType::q8) {
    if (const std::size_t smem = attention_smem(cfg_); smem > 48 * 1024)
        throw std::runtime_error("CudaTransformer: seq_len " + std::to_string(cfg_.seq_len) +
                                 " needs " + std::to_string(smem) +
                                 " bytes of attention scores, more than 48 KB");

    // The embedding stays fp32 either way, as on the CPU: a token reads only
    // its own row.
    auto proj = [&](const Tensor& t) {
        Proj p;
        if (quantized_) {
            p.q8 = DeviceQTensor(quantize_q8(t));
        } else {
            p.f32 = DeviceTensor(t);
        }
        return p;
    };
    const Weights& w = ckpt.weights;
    token_embedding_ = DeviceTensor(w.token_embedding);
    rms_final_ = upload(w.rms_final);
    wcls_ = proj(w.wcls);
    for (const LayerWeights& lw : w.layers) {
        layers_.push_back({upload(lw.rms_att), proj(lw.wqkv), proj(lw.wo), upload(lw.rms_ffn),
                           proj(lw.w13), proj(lw.w2)});
        key_cache_.emplace_back(cfg_.seq_len, cfg_.kv_dim());
        value_cache_.emplace_back(cfg_.seq_len, cfg_.kv_dim());
    }
    std::vector<float> cos, sin;
    rope_tables(cfg_, cos, sin);
    rope_cos_ = upload(cos);
    rope_sin_ = upload(sin);
    logits_dev_ = DeviceTensor(1, cfg_.vocab_size);
    logits_.resize(cfg_.vocab_size);
    const std::size_t feed_bytes = (cfg_.seq_len + 1) * sizeof(int);
    cuda_check(cudaMallocHost(&feed_host_, feed_bytes), "alloc pinned feed");
    cuda_check(cudaMalloc(&feed_dev_, feed_bytes), "alloc feed");
    cuda_check(cudaMallocHost(&logits_host_, cfg_.vocab_size * sizeof(float)),
               "alloc pinned logits");
    reserve_rows(1);
}

CudaTransformer::~CudaTransformer() {
    if (decode_graph_) cudaGraphExecDestroy(decode_graph_);
    cudaFreeHost(feed_host_);
    cudaFree(feed_dev_);
    cudaFreeHost(logits_host_);
}

void CudaTransformer::reserve_rows(std::size_t n) {
    if (n <= rows_) return;
    const std::size_t D = cfg_.dim, H = cfg_.hidden_dim;
    x_ = DeviceTensor(n, D);
    qkv_ = DeviceTensor(n, D + 2 * cfg_.kv_dim());
    att_ = DeviceTensor(n, D);
    gu_ = DeviceTensor(n, 2 * H);
    hb_ = DeviceTensor(n, H);
    rows_ = n;
    ++own_generation_;
}

void CudaTransformer::project(std::size_t n, const float* in, const Proj& w, float* out,
                              const MatmulFusion& fusion) const {
    if (quantized_) {
        matmul_cuda(n, w.q8.cols(), w.q8.rows(), in, w.q8.data(), w.q8.scale(), out,
                    GpuKernel::automatic, fusion);
    } else {
        matmul_cuda(n, w.f32.cols(), w.f32.rows(), in, w.f32.data(), out, GpuKernel::automatic,
                    fusion);
    }
}

std::size_t CudaTransformer::buffers_generation() const {
    return own_generation_ + cuda_scratch_generation();
}

const std::vector<float>& CudaTransformer::forward(int token, std::size_t pos) {
    return run(&token, 1, pos);
}

const std::vector<float>& CudaTransformer::prefill(const std::vector<int>& tokens,
                                                   std::size_t pos) {
    if (tokens.empty()) throw std::invalid_argument("prefill: no tokens");
    return run(tokens.data(), tokens.size(), pos);
}

const std::vector<float>& CudaTransformer::run(const int* tokens, std::size_t n,
                                               std::size_t pos) {
    check_feed(cfg_, tokens, n, pos, filled_);
    reserve_rows(n);
    feed_host_[0] = static_cast<int>(pos);
    std::memcpy(feed_host_ + 1, tokens, n * sizeof(int));

    const bool replay = n == 1 && decode_graph_ && graph_generation_ == buffers_generation();
    if (replay) {
        cuda_check(cudaGraphLaunch(decode_graph_, cudaStreamPerThread), "graph launch");
    } else {
        enqueue(n);
    }
    cuda_check(cudaStreamSynchronize(cudaStreamPerThread), "forward");
    filled_ = pos + n;
    std::copy(logits_host_, logits_host_ + cfg_.vocab_size, logits_.begin());

    // Capture after an eager one-token run, which has already grown every
    // buffer the step needs, so capturing allocates nothing.
    if (n == 1 && !replay) {
        if (decode_graph_) cuda_check(cudaGraphExecDestroy(decode_graph_), "destroy graph");
        decode_graph_ = nullptr;
        cudaGraph_t graph = nullptr;
        cuda_check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal),
                   "begin capture");
        try {
            enqueue(1);
        } catch (...) {
            cudaStreamEndCapture(cudaStreamPerThread, &graph);
            if (graph) cudaGraphDestroy(graph);
            throw;
        }
        cuda_check(cudaStreamEndCapture(cudaStreamPerThread, &graph), "end capture");
        cudaGraphExec_t exec = nullptr;
        const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
        cudaGraphDestroy(graph);
        cuda_check(err, "instantiate graph");
        decode_graph_ = exec;
        graph_generation_ = buffers_generation();
    }
    return logits_;
}

void CudaTransformer::enqueue(std::size_t n) {
    const std::size_t D = cfg_.dim, H = cfg_.hidden_dim, KV = cfg_.kv_dim();
    const std::size_t qkv_cols = D + 2 * KV;
    const int Di = int(D), Hi = int(H), KVi = int(KV), hs = int(cfg_.head_size());
    const int rows = int(n);
    cudaStream_t stream = cudaStreamPerThread;

    cuda_check(cudaMemcpyAsync(feed_dev_, feed_host_, (n + 1) * sizeof(int),
                               cudaMemcpyHostToDevice, stream),
               "upload feed");
    embed<<<rows, kThreads>>>(Di, feed_dev_, token_embedding_.data(), x_.data());
    check_launch();

    // Per layer: qkv = rmsnorm(x) Wqkv; RoPE and the cache write; attention;
    // x += att Wo; gu = rmsnorm(x) W13; x += swiglu(gu) W2. The norms and
    // residual adds are fused into the matmuls (MatmulFusion).
    const std::size_t smem = attention_smem(cfg_);
    const int group = int(cfg_.n_heads / cfg_.n_kv_heads);
    for (std::size_t l = 0; l < cfg_.n_layers; ++l) {
        Layer& lw = layers_[l];
        project(n, x_.data(), lw.wqkv, qkv_.data(), rms(lw.rms_att));
        rope_and_cache<<<dim3(ceil_div(D / 2 + KV / 2 + KV, kThreads), rows), kThreads>>>(
            int(qkv_cols), Di, KVi, hs, feed_dev_, rope_cos_.data(), rope_sin_.data(),
            qkv_.data(), key_cache_[l].data(), value_cache_[l].data());
        check_launch();
        attention<<<dim3(unsigned(cfg_.n_heads), rows), kThreads, smem>>>(
            int(qkv_cols), Di, KVi, hs, group, feed_dev_, qkv_.data(), key_cache_[l].data(),
            value_cache_[l].data(), att_.data());
        check_launch();
        project(n, att_.data(), lw.wo, x_.data(), residual());
        project(n, x_.data(), lw.w13, gu_.data(), rms(lw.rms_ffn));
        swiglu<<<ceil_div(n * H, kThreads), kThreads>>>(rows, Hi, gu_.data(), hb_.data());
        check_launch();
        project(n, hb_.data(), lw.w2, x_.data(), residual());
    }

    // Only the last row's logits are returned.
    project(1, x_.data() + (n - 1) * D, wcls_, logits_dev_.data(), rms(rms_final_));
    cuda_check(cudaMemcpyAsync(logits_host_, logits_dev_.data(), cfg_.vocab_size * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "download logits");
}
