
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>

#define MAX_ITERATIONS 10000
#define CONVERGENCE_THRESHOLD 1e-6

// Atomic max for float via bitwise reinterpret
__device__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// CUDA kernel to perform one iteration
__global__
void laplaceIterationKernel(double* grid, double* newGrid, int width, int height, float* maxDiff) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        double newVal = 0.25 * (grid[idx - 1] + grid[idx + 1] +
                                grid[idx - width] + grid[idx + width]);
        newGrid[idx] = newVal;

        float diff = fabsf(static_cast<float>(newVal - grid[idx]));
        atomicMaxFloat(maxDiff, diff);
    }
}

void initializeGrid(std::vector<double>& grid, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        grid[i] = 0.0;
    }

    for (int x = 0; x < width; x++) {
        grid[x] = 5.0;
        grid[(height - 1) * width + x] = -5.0;
    }
}

int main() {
    const int grid_sizes[] = {128, 256, 512, 1024};

    std::cout << std::left << std::setw(10) << "Grid Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Iterations" << "\n";
    std::cout << "----------------------------------------\n";

    std::ofstream results("cuda_laplace_results.csv");
    results << "GridSize,Time,Iterations\n";

    for (int size : grid_sizes) {
        int width = size;
        int height = size;
        int totalSize = width * height;

        std::vector<double> grid(totalSize, 0.0);
        std::vector<double> newGrid(totalSize, 0.0);
        initializeGrid(grid, width, height);

        double *d_grid, *d_newGrid;
        float *d_maxDiff;
        cudaMalloc(&d_grid, totalSize * sizeof(double));
        cudaMalloc(&d_newGrid, totalSize * sizeof(double));
        cudaMalloc(&d_maxDiff, sizeof(float));

        cudaMemcpy(d_grid, grid.data(), totalSize * sizeof(double), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

        float maxDiff;
        int iterations = 0;

        auto start = std::chrono::high_resolution_clock::now();

        do {
            maxDiff = 0.0f;
            cudaMemcpy(d_maxDiff, &maxDiff, sizeof(float), cudaMemcpyHostToDevice);

            laplaceIterationKernel<<<numBlocks, threadsPerBlock>>>(
                d_grid, d_newGrid, width, height, d_maxDiff);
            cudaDeviceSynchronize();

            std::swap(d_grid, d_newGrid);
            iterations++;

            cudaMemcpy(&maxDiff, d_maxDiff, sizeof(float), cudaMemcpyDeviceToHost);
        } while (maxDiff > CONVERGENCE_THRESHOLD && iterations < MAX_ITERATIONS);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << std::left << std::setw(10) << size
                  << std::setw(15) << std::fixed << std::setprecision(2) << duration.count()
                  << std::setw(15) << iterations << "\n";

        results << size << "," << duration.count() << "," << iterations << "\n";

        if (size == grid_sizes[0]) {
            cudaMemcpy(grid.data(), d_grid, totalSize * sizeof(double), cudaMemcpyDeviceToHost);
            std::ofstream file("cuda_laplace_grid.csv");
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    file << grid[y * width + x];
                    if (x < width - 1) file << ",";
                }
                file << "\n";
            }
            file.close();
        }

        cudaFree(d_grid);
        cudaFree(d_newGrid);
        cudaFree(d_maxDiff);
    }

    results.close();
    return 0;
}
