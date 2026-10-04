#include <iostream>
#include <chrono>
#include "tensor.h"

int main() {
    // 1) Correctness check
    Tensor A(2, 3), B(3, 2);
    float v = 1.0f;
    for (auto& x : A.data) x = v++;   // A = [[1,2,3],[4,5,6]]
    for (auto& x : B.data) x = v++;   // B = [[7,8],[9,10],[11,12]]

    Tensor C = matmul(A, B);          // expected: [[58,64],[139,154]]
    for (std::size_t i = 0; i < C.rows; ++i) {
        for (std::size_t j = 0; j < C.cols; ++j) std::cout << C.at(i, j) << " ";
        std::cout << "\n";
    }

    // 2) Benchmark
    const std::size_t N = 512;
    Tensor X(N, N), Y(N, N);
    for (auto& x : X.data) x = 1.0f;
    for (auto& y : Y.data) y = 1.0f;

    auto start = std::chrono::high_resolution_clock::now();
    Tensor Z = matmul(X, Y);
    auto end = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(end - start).count();
    double gflops = (2.0 * N * N * N) / secs / 1e9;
    std::cout << N << "x" << N << " matmul: " << secs * 1000 << " ms, "
              << gflops << " GFLOPS\n";
}