# Sobel Edge Detection Project

This project implements the Sobel edge detection algorithm in three versions:
1. **CPU Implementation** (`sobel_cpu.cpp`)
2. **Naive GPU Implementation** (`sobel_gpu_naive.cu`)
3. **Optimized GPU Implementation** (`sobel_gpu_optimized.cu`)

The GPU implementations use CUDA to accelerate the computation.

---

## Prerequisites

1. **CUDA Toolkit**: Ensure the CUDA Toolkit is installed on your system.
   - Verify installation:
     ```bash
     nvcc --version
     ```

2. **OpenCV**: Install OpenCV for image processing.
   - On Ubuntu:
     ```bash
     sudo apt-get install libopencv-dev
     ```
   - Or, use `vcpkg`:
     ```bash
     ./vcpkg install opencv
     ```

3. **Input Image**: Place an input image named `input_image.jpg` in the project directory.

---

## Compilation and Execution

### 1. **CPU Implementation**
Compile and run the CPU version:
```bash
g++ sobel_cpu.cpp -o sobel_cpu `pkg-config --cflags --libs opencv4`
./sobel_cpu
```
- Output: `cpu_output_image.jpg`

---

### 2. **Naive GPU Implementation**
Compile and run the naive GPU version:
```bash
nvcc sobel_gpu_naive.cu -o sobel_gpu_naive `pkg-config --cflags --libs opencv4` -lineinfo
./sobel_gpu_naive
```
- Output: `gpu_output_image_naive.jpg`

---

### 3. **Optimized GPU Implementation**
Compile and run the optimized GPU version:
```bash
nvcc sobel_gpu_optimized.cu -o sobel_gpu_optimized `pkg-config --cflags --libs opencv4` -lineinfo
./sobel_gpu_optimized
```
- Output: `gpu_output_image_optimized.jpg`

---

## Profiling with Nsight Compute

To analyze the performance of the GPU implementations, use **NVIDIA Nsight Compute**:

1. **Naive GPU Version**:
   ```bash
   nv-nsight-cu-cli ./sobel_gpu_naive
   ```

2. **Optimized GPU Version**:
   ```bash
   nv-nsight-cu-cli ./sobel_gpu_optimized
   ```

For a graphical interface, launch Nsight Compute:
```bash
nv-nsight-cu
```

---

## Expected Outputs

- **CPU Output**: `cpu_output_image.jpg`
- **Naive GPU Output**: `gpu_output_image_naive.jpg`
- **Optimized GPU Output**: `gpu_output_image_optimized.jpg`

Compare the outputs visually to validate the correctness of the edge detection algorithm.

---

## Notes

- Ensure your system has a CUDA-capable GPU.
- If running in a headless environment, use `xvfb` for OpenCV GUI functions:
  ```bash
  sudo apt-get install xvfb
  xvfb-run ./sobel_gpu_naive
  ```

---