#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/cuda.h> // For CUDA operations
#include <torch/script.h>

#define AIFILTER_IMG 320