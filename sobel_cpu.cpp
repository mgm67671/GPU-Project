#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp> // OpenCV for image handling

// Function to apply the Sobel filter
void applySobel(const cv::Mat& input, cv::Mat& output) {
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

    // Initialize the output image
    output = cv::Mat::zeros(input.size(), CV_8UC1);

    // Apply the Sobel filter
    for (int y = 1; y < input.rows - 1; ++y) {
        for (int x = 1; x < input.cols - 1; ++x) {
            int sumX = 0;
            int sumY = 0;

            // Convolution with Sobel kernels
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixel = input.at<uchar>(y + ky, x + kx);
                    sumX += pixel * Gx[ky + 1][kx + 1];
                    sumY += pixel * Gy[ky + 1][kx + 1];
                }
            }

            // Compute gradient magnitude
            int magnitude = std::sqrt(sumX * sumX + sumY * sumY);
            magnitude = std::min(255, magnitude); // Clamp to 255

            output.at<uchar>(y, x) = static_cast<uchar>(magnitude);
        }
    }
}

int main() {
    // Load the input image in grayscale
    cv::Mat input = cv::imread("input_image.jpg", cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cerr << "Error: Could not load input image!" << std::endl;
        return -1;
    }

    // Output image
    cv::Mat output;

    // Apply the Sobel filter
    applySobel(input, output);

    // Save the result
    cv::imwrite("cpu_output_image.jpg", output);
    // cv::imshow("Input Image", input);
    // cv::imshow("Sobel Edge Detection", output);
    // cv::waitKey(0);

    return 0;
}