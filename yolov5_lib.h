
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


// void get_input_data_cv(const cv::Mat& sample, uint8_t* input_data, int img_h, int img_w, float input_scale, int zero_point, int swapRB = 0);

// layer make_darknet_layer(int batch, int w, int h, int net_w, int net_h, int n, int total, int classes, int layer_type);

// void free_darknet_layer(layer l);

// static int entry_index(layer l, int batch, int location, int entry);

// void logistic_cpu(float* input, int size);

// int yolo_num_detections(layer l, float thresh);

// // int num_detections(vector<layer> layers_params, float thresh);

// // detection* make_network_boxes(vector<layer> layers_params, float thresh, int* num);

// void correct_yolo_boxes(detection* dets, int n, int w, int h, int netw, int neth, int relative);

// box get_yolo_box(float* x, float* biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride);

// int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int* map, int relative,
//                         detection* dets);

// void correct_region_boxes(detection* dets, int n, int w, int h, int netw, int neth, int relative);

// box get_region_box(float* x, float* biases, int n, int index, int i, int j, int w, int h, int stride);

// void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int* map, float tree_thresh,
//                            int relative, detection* dets);

// // void fill_network_boxes(vector<layer> layers_params, int img_w, int img_h, int net_w, int net_h, float thresh,
// //                         float hier, int* map, int relative, detection* dets);

// // detection* get_network_boxes(vector<layer> layers_params, int img_w, int img_h, int net_w, int net_h, float thresh,
// //                              float hier, int* map, int relative, int* num);

// void free_detections(detection* dets, int nboxes);

// float overlap(float x1, float w1, float x2, float w2);

// float box_intersection(box a, box b);

// float box_union(box a, box b);

// float box_iou(box a, box b);

// void do_nms_sort(detection* dets, int total, int classes, float thresh);

// void show_usage();

// int check_file_exist(const char* file_name);

// static cv::Scalar obj_id_to_color(int obj_id);

// void postpress_graph(cv::Mat& frame, graph_t graph, int output_node_num,int net_w, int net_h,int numBBoxes,int total_numAnchors,int layer_type,char *window_name);

// void thread_camera();

void postpress_graph_image_wrapper(void* data_pointer, int height, int width, float* array, graph_t graph, int output_node_num,int net_w, int net_h,int numBBoxes,int total_numAnchors,int layer_type);

void postpress_graph_image(const cv::Mat& sample, graph_t graph, int output_node_num,int net_w, int net_h,int numBBoxes,int total_numAnchors,int layer_type);

int set_graph(int net_h, int net_w, graph_t graph);

int set_image(const char* image_file, tensor_t input_tensor, int net_h, int net_w);

int set_image_wrapper(void* data_pointer, int height, int width, tensor_t input_tensor, int net_h, int net_w);

int inference(const char* image_file, const char* model_file, context_t timvx_context, int rtt);
}