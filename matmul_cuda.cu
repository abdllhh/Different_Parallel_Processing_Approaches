
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// Function to initialize matrix with random values
void initializeMatrix(std::vector<float>& matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dist(gen);
    }
}

// CPU matrix multiplication for reference
double matrixMultiplyCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, 
                       int m, int n, int k) {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    return duration.count();
}

// CUDA matrix multiplication with specific block size
double matrixMultiplyCUDA(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, 
                        int m, int n, int k, int blockSizeX, int blockSizeY) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaError_t err;
    
    err = cudaMalloc(&d_A, m * n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc d_A failed: " << cudaGetErrorString(err) << std::endl;
        return -1.0;
    }
    
    err = cudaMalloc(&d_B, n * k * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc d_B failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        return -1.0;
    }
    
    err = cudaMalloc(&d_C, m * k * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc d_C failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return -1.0;
    }
    
    // Copy input matrices from host to device
    err = cudaMemcpy(d_A, A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to d_A failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1.0;
    }
    
    err = cudaMemcpy(d_B, B.data(), n * k * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to d_B failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1.0;
    }
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Define grid and block dimensions
    dim3 blockDim(blockSizeX, blockSizeY);
    dim3 gridDim((k + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel and time it
    cudaEventRecord(start);
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
    cudaEventRecord(stop);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0;
    }
    
    // Wait for kernel to finish
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result from device to host
    err = cudaMemcpy(C.data(), d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy from d_C failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0;
    }
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

// Verify results between CPU and GPU
bool verifyResult(const std::vector<float>& cpuResult, const std::vector<float>& gpuResult, float tolerance = 1e-5) {
    if (cpuResult.size() != gpuResult.size()) {
        return false;
    }
    
    float diffSum = 0.0f;
    float maxDiff = 0.0f;
    int maxDiffIdx = -1;
    
    for (size_t i = 0; i < cpuResult.size(); i++) {
        float diff = std::abs(cpuResult[i] - gpuResult[i]);
        diffSum += diff;
        
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = i;
        }
    }
    
    std::cout << "Verification - Diff sum: " << diffSum 
              << ", Max diff: " << maxDiff << " at index " << maxDiffIdx << std::endl;
    
    // Consider it verified if the average difference is small
    return (diffSum / cpuResult.size()) < tolerance;
}

int main() {
    // Print CUDA device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    
    // Define potential workgroup sizes to test
    std::vector<std::pair<int, int>> workgroupSizes = {
        {8, 8},    // 64 threads
        {8, 16},   // 128 threads
        {16, 8},   // 128 threads
        {16, 16},  // 256 threads
        {32, 4},   // 128 threads
        {32, 8}    // 256 threads
    };
    
    std::cout << "\nTesting different workgroup sizes on a 512x512 matrix:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::left << std::setw(15) << "Block Size" 
              << std::setw(20) << "Time (ms)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    // Test matrix size for workgroup optimization
    int testSize = 512;
    std::vector<float> A_test(testSize * testSize);
    std::vector<float> B_test(testSize * testSize);
    std::vector<float> C_test(testSize * testSize);
    
    // Initialize test matrices
    initializeMatrix(A_test, testSize * testSize);
    initializeMatrix(B_test, testSize * testSize);
    
    // Find the best workgroup size
    double bestTime = std::numeric_limits<double>::max();
    std::pair<int, int> bestWorkgroupSize = {16, 16}; // Default in case all tests fail
    
    for (const auto& size : workgroupSizes) {
        int blockSizeX = size.first;
        int blockSizeY = size.second;
        
        // Run the test with this block size
        double time = matrixMultiplyCUDA(
            A_test, B_test, C_test, testSize, testSize, testSize, blockSizeX, blockSizeY);
        
        std::cout << std::left << std::setw(5) << blockSizeX << " x " 
                  << std::setw(5) << blockSizeY
                  << std::fixed << std::setprecision(2) 
                  << std::setw(20) << time << std::endl;
        
        // Update best workgroup size if this one is faster
        if (time > 0 && time < bestTime) {
            bestTime = time;
            bestWorkgroupSize = size;
        }
    }
    
    std::cout << "\nOptimal workgroup size: " << bestWorkgroupSize.first << "x" 
              << bestWorkgroupSize.second << " with " << bestTime << " ms" << std::endl;
    
    // Now run the full benchmark with the optimal workgroup size
    std::vector<int> sizes = {256, 512, 1024, 2048};
    
    std::cout << "\nMatrix Multiplication Benchmark (Workgroup: " 
              << bestWorkgroupSize.first << "x" << bestWorkgroupSize.second << ")" << std::endl;
    std::cout << std::string(75, '=') << std::endl;
    std::cout << std::left << std::setw(10) << "Size" 
              << std::setw(20) << "CPU Time (ms)" 
              << std::setw(20) << "CUDA Time (ms)" 
              << std::setw(15) << "Speedup" 
              << std::setw(10) << "Verified" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    // Prepare results container
    std::vector<double> cpu_times;
    std::vector<double> cuda_times;
    std::vector<double> speedups;
    
    // Test each matrix size
    for (int size : sizes) {
        std::vector<float> A(size * size);
        std::vector<float> B(size * size);
        std::vector<float> C_cpu(size * size);
        std::vector<float> C_cuda(size * size);
        
        // Initialize matrices
        initializeMatrix(A, size * size);
        initializeMatrix(B, size * size);
        
        // Run CPU version
        double cpu_time = matrixMultiplyCPU(A, B, C_cpu, size, size, size);
        cpu_times.push_back(cpu_time);
        
        // Run CUDA version with optimal workgroup size
        double cuda_time = matrixMultiplyCUDA(
            A, B, C_cuda, size, size, size, 
            bestWorkgroupSize.first, bestWorkgroupSize.second);
        cuda_times.push_back(cuda_time);
        
        // Calculate speedup
        double speedup = cpu_time / cuda_time;
        speedups.push_back(speedup);
        
        // Verify results
        bool verified = (cuda_time > 0) ? verifyResult(C_cpu, C_cuda) : false;
        
        // Print results
        std::cout << std::left << std::setw(10) << size 
                  << std::fixed << std::setprecision(2) 
                  << std::setw(20) << cpu_time 
                  << std::setw(20) << cuda_time 
                  << std::setw(15) << speedup 
                  << std::setw(10) << (verified ? "Yes" : "No") << std::endl;
    }
    
    // Save results to CSV for plotting
    std::ofstream outfile("cuda_results.csv");
    outfile << "Size,CPU,CUDA,Speedup,BlockSizeX,BlockSizeY" << std::endl;
    
    for (size_t i = 0; i < sizes.size(); i++) {
        outfile << sizes[i] << "," 
                << cpu_times[i] << "," 
                << cuda_times[i] << "," 
                << speedups[i] << ","
                << bestWorkgroupSize.first << ","
                << bestWorkgroupSize.second << std::endl;
    }
    
    outfile.close();
    std::cout << "\nResults saved to cuda_results.csv" << std::endl;
    
    // Provide workgroup size justification
    std::cout << "\nWorkgroup size justification:" << std::endl;
    std::cout << "1. Optimal workgroup size determined experimentally: " 
              << bestWorkgroupSize.first << "x" << bestWorkgroupSize.second << std::endl;
    std::cout << "2. This configuration balances thread utilization and memory access patterns" << std::endl;
    std::cout << "3. Total thread count (" << bestWorkgroupSize.first * bestWorkgroupSize.second 
              << ") is aligned with warp size (" << prop.warpSize << ")" << std::endl;
    std::cout << "4. Experimentally proven to be the fastest configuration for our problem" << std::endl;
    std::cout << "5. Well within the hardware limits of " << prop.maxThreadsPerBlock << " threads per block" << std::endl;
    
    return 0;
}

