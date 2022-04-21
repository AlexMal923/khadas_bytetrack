
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <vector>

#ifdef _MSC_VER  
#define NOMINMAX
#endif

#include <algorithm>
#include <cmath>
#include <string>

#include <pthread.h>
#include <sys/time.h>
#include <sched.h>
#include <sys/resource.h>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>

#include "common.h"
#include "tengine/c_api.h"

extern "C" {

typedef struct
{
    float x, y, w, h;
} box;

typedef struct
{
    box bbox;
    int classes;
    float* prob;
    float* mask;
    float objectness;
    int sort_class;
} detection;

typedef struct layer
{
    int layer_type;
    int batch;
    int total;
    int n, c, h, w;
    int out_n, out_c, out_h, out_w;
    int classes;
    int inputs;
    int outputs;
    int* mask;
    float* biases;
    float* output;
    int coords;
} layer;

void postpress_graph_image_wrapper(void* data_pointer, int height, int width, float* array, graph_t graph, int output_node_num,int net_w, int net_h,int numBBoxes,int total_numAnchors,int layer_type);

void postpress_graph_image(const cv::Mat& sample, graph_t graph, int output_node_num,int net_w, int net_h,int numBBoxes,int total_numAnchors,int layer_type);

int set_graph(int net_h, int net_w, graph_t graph);

int set_image(const char* image_file, tensor_t input_tensor, int net_h, int net_w);

int set_image_wrapper(void* data_pointer, int height, int width, tensor_t input_tensor, int net_h, int net_w);

int inference(const char* image_file, const char* model_file, context_t timvx_context, int rtt);
}
