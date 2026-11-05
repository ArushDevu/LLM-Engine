#pragma once
#include <cstddef>

// Causal attention over a KV cache on an NVIDIA GPU. Built only with
// -DENGINE_CUDA=ON. Runs on the calling thread's default stream and returns
// before the GPU finishes, like matmul_cuda().

struct AttentionShape {
    int n_heads, n_kv_heads, head_size, seq_len;
    std::size_t q_stride;    // floats between rows of q; head h starts at h * head_size
    std::size_t kv_stride;   // floats between positions of the caches (kv_dim)
    std::size_t out_stride;  // floats between rows of out
};

enum class AttentionKernel {
    automatic,     // split for few rows, single_block for many
    single_block,  // one block per (head, row); scores for the whole context in shared memory
    split,         // positions split across blocks; the last block to finish merges them
};

// Floats of scratch attention_cuda() needs for rows rows (0 for single_block).
std::size_t attention_scratch_floats(const AttentionShape& s, std::size_t rows,
                                     AttentionKernel kernel = AttentionKernel::automatic);

// Throws std::invalid_argument if a kernel can't handle s: single_block keeps
// head_size + seq_len floats in shared memory, so seq_len is capped at ~12k;
// split keeps only a chunk of positions and has no practical cap.
void attention_check(const AttentionShape& s, AttentionKernel kernel = AttentionKernel::automatic);

// For each of rows rows and each query head h: row i sits at position
// *pos_dev + i and attends to cached positions [0, *pos_dev + i], with kv head
// h / (n_heads / n_kv_heads). out[i][h * head_size + d] = sum over t of
// softmax(q . k_t / sqrt(head_size)) v_t[d]. The position is read from device
// memory, so a CUDA graph captured once serves every position. scratch holds
// attention_scratch_floats(s, rows, kernel) floats and must be zero-filled
// before its first use; calls leave it ready for the next.
void attention_cuda(const AttentionShape& s, std::size_t rows, const int* pos_dev,
                    const float* q, const float* key_cache, const float* value_cache, float* out,
                    float* scratch, AttentionKernel kernel = AttentionKernel::automatic);
