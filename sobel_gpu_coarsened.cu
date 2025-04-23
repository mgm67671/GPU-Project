#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Define the coarsening factor as a global constant
constexpr int COARSENING_FACTOR = 2;

// CUDA kernel for applying the Sobel filter with thread coarsening
__global__ void sobelKernelCoarsened(const unsigned char* input, unsigned char* output, int width, int height) 
{
    // Shared memory for the tile
    __shared__ unsigned char sharedMem[16 * COARSENING_FACTOR + 2][16 * COARSENING_FACTOR + 2]; // Adjusted for block size and borders

    // Calculate global thread coordinates
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * COARSENING_FACTOR; // Each thread processes COARSENING_FACTOR x COARSENING_FACTOR pixels
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * COARSENING_FACTOR;

    // Calculate shared memory coordinates
    int sharedX = threadIdx.x * COARSENING_FACTOR + 1;
    int sharedY = threadIdx.y * COARSENING_FACTOR + 1;

    // Load data into shared memory (each thread loads a COARSENING_FACTOR x COARSENING_FACTOR block)
    for (int dy = 0; dy < COARSENING_FACTOR; ++dy) 
    {
        for (int dx = 0; dx < COARSENING_FACTOR; ++dx) 
        {
            int globalX = x + dx;
            int globalY = y + dy;
            if (globalX < width && globalY < height)
                sharedMem[sharedY + dy][sharedX + dx] = input[globalY * width + globalX];
        }
    }

    // Load border pixels (similar logic as before, adjusted for coarsening)
    if (threadIdx.x == 0 && x > 0) 
    {
        for (int dy = 0; dy < COARSENING_FACTOR; ++dy) 
        {
            if (y + dy < height)
                sharedMem[sharedY + dy][0] = input[(y + dy) * width + (x - 1)];
        }
    }
    if (threadIdx.x == blockDim.x - 1 && x + COARSENING_FACTOR < width) 
    {
        for (int dy = 0; dy < COARSENING_FACTOR; ++dy) 
        {
            if (y + dy < height)
                sharedMem[sharedY + dy][sharedX + COARSENING_FACTOR] = input[(y + dy) * width + (x + COARSENING_FACTOR)];
        }
    }
    if (threadIdx.y == 0 && y > 0) 
    {
        for (int dx = 0; dx < COARSENING_FACTOR; ++dx) 
        {
            if (x + dx < width)
                sharedMem[0][sharedX + dx] = input[(y - 1) * width + (x + dx)];
        }
    }
    if (threadIdx.y == blockDim.y - 1 && y + COARSENING_FACTOR < height) 
    {
        for (int dx = 0; dx < COARSENING_FACTOR; ++dx) 
        {
            if (x + dx < width)
                sharedMem[sharedY + COARSENING_FACTOR][sharedX + dx] = input[(y + COARSENING_FACTOR) * width + (x + dx)];
        }
    }

    // Synchronize threads to ensure shared memory is fully loaded
    __syncthreads();

    // Sobel kernels
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    // Apply Sobel filter to each pixel in the COARSENING_FACTOR x COARSENING_FACTOR block
    for (int dy = 0; dy < COARSENING_FACTOR; ++dy) 
    {
        for (int dx = 0; dx < COARSENING_FACTOR; ++dx) 
        {
            int globalX = x + dx;
            int globalY = y + dy;
            if (globalX > 0 && globalX < width - 1 && globalY > 0 && globalY < height - 1) 
            {
                int sumX = 0, sumY = 0;

                // Convolution with Sobel kernels
                for (int ky = -1; ky <= 1; ++ky) 
                {
                    for (int kx = -1; kx <= 1; ++kx) 
                    {
                        int pixel = sharedMem[sharedY + dy + ky][sharedX + dx + kx];
                        sumX += pixel * Gx[ky + 1][kx + 1];
                        sumY += pixel * Gy[ky + 1][kx + 1];
                    }
                }

                // Compute gradient magnitude
                int magnitude = sqrtf(sumX * sumX + sumY * sumY);
                magnitude = min(255, magnitude); // Clamp to 255

                // Write the result to the output image
                output[globalY * width + globalX] = static_cast<unsigned char>(magnitude);
            }
        }
    }
}

int main() 
{
    // Load the input image in grayscale
    cv::Mat input = cv::imread("input_image.jpg", cv::IMREAD_GRAYSCALE);
    if (input.empty()) 
    {
        std::cerr << "Error: Could not load input image!" << std::endl;
        return -1;
    }

    int width = input.cols;
    int height = input.rows;

    // Allocate memory for the output image
    cv::Mat output(height, width, CV_8UC1);

    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    // Copy input image to device
    cudaMemcpy(d_input, input.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid sizes (adjusted for 16x16 block size with COARSENING_FACTOR)
    dim3 blockSize(16, 16); // Each thread processes COARSENING_FACTOR x COARSENING_FACTOR pixels
    dim3 gridSize((width + blockSize.x * COARSENING_FACTOR - 1) / (blockSize.x * COARSENING_FACTOR), 
                  (height + blockSize.y * COARSENING_FACTOR - 1) / (blockSize.y * COARSENING_FACTOR));

    // Launch the coarsened Sobel kernel
    sobelKernelCoarsened<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Copy the result back to the host
    cudaMemcpy(output.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save the result
    cv::imwrite("gpu_output_image_coarsened.jpg", output);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}