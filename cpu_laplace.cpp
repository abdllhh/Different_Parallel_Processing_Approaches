#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>

// Parameters for Laplace solver
const int MAX_ITERATIONS = 10000;
const double CONVERGENCE_THRESHOLD = 1e-6;

// Function to initialize grid with boundary conditions
void initializeGrid(std::vector<double>& grid, int width, int height) {
    // Initialize all points to 0
    for (int i = 0; i < width * height; i++) {
        grid[i] = 0.0;
    }
    
    // Set boundary conditions:
    // Top boundary (row 0) = 5V
    for (int x = 0; x < width; x++) {
        grid[x] = 5.0;
    }
    
    // Bottom boundary (row height-1) = -5V
    for (int x = 0; x < width; x++) {
        grid[(height - 1) * width + x] = -5.0;
    }
    
    // Left and right boundaries are kept at 0V
    // This is already handled by the initial grid initialization
}

// Function to solve Laplace equation using Jacobi iteration
int solveLaplaceCPU(std::vector<double>& grid, std::vector<double>& newGrid, int width, int height) {
    double maxDiff;
    int iterations = 0;
    
    do {
        maxDiff = 0.0;
        
        // Update all non-boundary points
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = y * width + x;
                
                // The new value is the average of the four adjacent values
                newGrid[idx] = 0.25 * (
                    grid[idx - 1] +      // Left neighbor
                    grid[idx + 1] +      // Right neighbor
                    grid[idx - width] +  // Top neighbor
                    grid[idx + width]    // Bottom neighbor
                );
                
                // Compute max difference for convergence check
                double diff = std::abs(newGrid[idx] - grid[idx]);
                maxDiff = std::max(maxDiff, diff);
            }
        }
        
        // Copy new values to grid (excluding boundaries)
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = y * width + x;
                grid[idx] = newGrid[idx];
            }
        }
        
        iterations++;
        
    } while (maxDiff > CONVERGENCE_THRESHOLD && iterations < MAX_ITERATIONS);
    
    return iterations;
}

// Function to save grid to a CSV file for visualization
void saveGridToCSV(const std::vector<double>& grid, int width, int height, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            file << grid[y * width + x];
            if (x < width - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "Grid saved to " << filename << std::endl;
}

int main() {
    // Grid size parameters
    const int grid_sizes[] = {128, 256, 512, 1024};
    
    std::cout << "CPU Laplace Solver" << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::left << std::setw(10) << "Grid Size" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Iterations" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Open file for results
    std::ofstream results("cpu_laplace_results.csv");
    results << "GridSize,Time,Iterations" << std::endl;
    
    // Test with different grid sizes
    for (int size : grid_sizes) {
        int width = size;
        int height = size;
        
        // Allocate memory for grids
        std::vector<double> grid(width * height, 0.0);
        std::vector<double> newGrid(width * height, 0.0);
        
        // Initialize grid with boundary conditions
        initializeGrid(grid, width, height);
        
        // Solve Laplace equation and measure time
        auto start = std::chrono::high_resolution_clock::now();
        
        int iterations = solveLaplaceCPU(grid, newGrid, width, height);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        
        // Output results
        std::cout << std::left << std::setw(10) << size 
                  << std::setw(15) << std::fixed << std::setprecision(2) << duration.count() 
                  << std::setw(15) << iterations << std::endl;
        
        // Save results to file
        results << size << "," << duration.count() << "," << iterations << std::endl;
        
        // Save the grid for the smallest size for visualization
        if (size == grid_sizes[0]) {
            saveGridToCSV(grid, width, height, "cpu_laplace_grid.csv");
        }
    }
    
    results.close();
    
    return 0;
}
