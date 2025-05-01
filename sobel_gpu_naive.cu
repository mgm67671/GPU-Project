#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// CUDA kernel for applying the Sobel filter
__global__ void sobelKernel(const unsigned char* input, unsigned char* output, int width, int height) 
{
    // Calculate the thread's position in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

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

    // Ensure the thread is within bounds and not on the border
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) 
    {
        int sumX = 0;
        int sumY = 0;

        // Apply the Sobel filter
        for (int ky = -1; ky <= 1; ++ky) 
        {
            for (int kx = -1; kx <= 1; ++kx) 
            {
                int pixel = input[(y + ky) * width + (x + kx)];
                sumX += pixel * Gx[ky + 1][kx + 1];
                sumY += pixel * Gy[ky + 1][kx + 1];
            }
        }

        // Compute gradient magnitude
        int magnitude = sqrtf(sumX * sumX + sumY * sumY);
        magnitude = min(255, magnitude); // Clamp to 255

        // Write the result to the output image
        output[y * width + x] = static_cast<unsigned char>(magnitude);
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

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the Sobel kernel
    sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Copy the result back to the host
    cudaMemcpy(output.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save the result
    cv::imwrite("gpu_output_image_naive.jpg", output);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}