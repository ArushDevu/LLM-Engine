#include <iostream>
#include <chrono>
#include "matmul.h"

int main() {
    const std::size_t N = 512;
    Tensor X(N, N), Y(N, N);
    for (auto& x : X.data) x = 1.0f;
    for (auto& y : Y.data) y = 1.0f;

    auto start = std::chrono::high_resolution_clock::now();
    Tensor Z = matmul_naive(X, Y);
    auto end = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(end - start).count();
    double gflops = (2.0 * N * N * N) / secs / 1e9;
    std::cout << N << "x" << N << " matmul: " << secs * 1000 << " ms, "
              << gflops << " GFLOPS\n";
}
