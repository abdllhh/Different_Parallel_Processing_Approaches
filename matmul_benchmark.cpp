%%writefile matmul_benchmark.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <omp.h>

// Function to initialize matrix with random values
void initializeMatrix(std::vector<float>& matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dist(gen);
    }
}

// Single-threaded CPU matrix multiplication
double matrixMultiplyCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, 
                       int m, int n, int k) {
    // Reset result matrix
    for (int i = 0; i < m * k; i++) {
        C[i] = 0.0f;
    }
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform matrix multiplication (naive/simple algorithm)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    return duration.count();
}

// OpenMP matrix multiplication
double matrixMultiplyOMP(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, 
                        int m, int n, int k) {
    // Reset result matrix
    #pragma omp parallel for
    for (int i = 0; i < m * k; i++) {
        C[i] = 0.0f;
    }
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform matrix multiplication with OpenMP
    #pragma omp parallel for
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

int main() {
    // Define matrix sizes to test
    std::vector<int> sizes = {256, 512, 1024, 2048};
    
    // Prepare results container
    std::vector<double> cpu_times;
    std::vector<double> omp_times;
    std::vector<double> speedup_omp;
    
    std::cout << "Matrix Multiplication Benchmark" << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << std::left << std::setw(10) << "Size" 
              << std::setw(20) << "CPU Time (ms)" 
              << std::setw(20) << "OpenMP Time (ms)" 
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    // Test each matrix size
    for (int size : sizes) {
        int m = size;  // Rows in A
        int n = size;  // Columns in A / Rows in B
        int k = size;  // Columns in B
        
        // Allocate memory for matrices
        std::vector<float> A(m * n);
        std::vector<float> B(n * k);
        std::vector<float> C_cpu(m * k);
        std::vector<float> C_omp(m * k);
        
        // Initialize matrices with random values
        initializeMatrix(A, m * n);
        initializeMatrix(B, n * k);
        
        // Run CPU single-threaded version
        double cpu_time = matrixMultiplyCPU(A, B, C_cpu, m, n, k);
        cpu_times.push_back(cpu_time);
        
        // Run OpenMP version
        double omp_time = matrixMultiplyOMP(A, B, C_omp, m, n, k);
        omp_times.push_back(omp_time);
        
        // Calculate speedup
        double speedup = cpu_time / omp_time;
        speedup_omp.push_back(speedup);
        
        // Print results
        std::cout << std::left << std::setw(10) << size 
                  << std::fixed << std::setprecision(2) 
                  << std::setw(20) << cpu_time 
                  << std::setw(20) << omp_time 
                  << std::setw(15) << speedup << std::endl;
    }
    
    // Write results to a CSV file for plotting
    std::ofstream outfile("cpu_omp_results.csv");
    outfile << "Size,CPU,OpenMP,Speedup_OMP" << std::endl;
    
    for (size_t i = 0; i < sizes.size(); i++) {
        outfile << sizes[i] << "," 
                << cpu_times[i] << "," 
                << omp_times[i] << "," 
                << speedup_omp[i] << std::endl;
    }
    
    outfile.close();
    
    std::cout << "\nResults saved to cpu_omp_results.csv" << std::endl;
    
    return 0;
}

