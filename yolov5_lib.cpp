/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "yolov5_lib.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

#define VXDEVICE  "TIMVX"

#define MODEL_COLS 352
#define MODEL_ROWS 352

#define DEBUG_OPTION 0

#define CAMERA_WEIGHT 640
#define CAMERA_HEIGHT 480

#if DEBUG_OPTION
static unsigned int tmpVal;
#endif
using namespace std;

const int classes = 80;
const float thresh = 0.5;
const float hier_thresh = 0.5;
const float nms = 0.45;
const int numBBoxes = 5;
const int relative = 1;
const int yolov3_numAnchors = 6;
const int yolov2_numAnchors = 5;


int num_thread = DEFAULT_THREAD_COUNT;

 
void get_input_data_cv(const cv::Mat& sample, uint8_t* input_data, int img_h, int img_w, float input_scale, int zero_point, int swapRB = 0)
{
	cv::Mat img;
	if (sample.channels() == 4)
	{
		cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
	}
	else if (sample.channels() == 1)
	{
		cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
	}
	else if (sample.channels() == 3 && swapRB == 1)
	{
		cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
        cout << "BGR2RGB!" << endl;
	}
	else
	{
		img = sample;
	}
	cv::Mat img2(img_h,img_w,CV_8UC3,cv::Scalar(0,0,0));
	int h,w;
	if((float)img.cols/(float)img_w > (float)img.rows/(float)img_h){
		h=1.0*img_w/img.cols*img.rows;
		w=img_w;
		cv::resize(img, img, cv::Size(w,h));
	}else{
		w=1.0*img_h/img.rows*img.cols;
		h=img_h;
		cv::resize(img, img, cv::Size(w,h));
	}
	int top = (img_h - h)/2;
	int bat = (img_h - h + 1)/2;
	int left = (img_w - w)/2;
	int right = (img_w - w + 1)/2;


	cv::copyMakeBorder(img,img2,top,bat,left,right,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
	uint8_t* img_data = img2.data;
	int hw = img_h * img_w;

	for (int h = 0; h < img_h; h++)
	{
		for (int w = 0; w < img_w; w++)
		{
			for (int c = 0; c < 3; c++)
			{
				int udata = *img_data;
				if (udata > 255)
					udata = 255;
				else if (udata < 0)
					udata = 0;
				input_data[c * hw + h * img_w + w] = udata;
				img_data++;
			}
		}
	}
}


layer make_darknet_layer(int batch, int w, int h, int net_w, int net_h, int n, int total, int classes, int layer_type)
{
    layer l = {0};
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n * (classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.inputs = l.w * l.h * l.c;
    // yolov5
    float biases[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
    // tiny
    float biases_tiny[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
    // yolov2
    float biases_yolov2[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};

    l.biases = ( float* )calloc(total * 2, sizeof(float));
    if (layer_type == 0)
    {
        l.mask = ( int* )calloc(n, sizeof(int));
        if (9 == total)
        {
            for (int i = 0; i < total * 2; ++i)
            {
                l.biases[i] = biases[i];
            }
            if (l.w == net_w / 32)
            {
                int j = 6;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
            if (l.w == net_w / 16)
            {
                int j = 3;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
            if (l.w == net_w / 8)
            {
                int j = 0;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
        }
        if (6 == total)
        {
            for (int i = 0; i < total * 2; ++i)
            {
                l.biases[i] = biases_tiny[i];
            }
            if (l.w == net_w / 32)
            {
                int j = 3;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
            if (l.w == net_w / 16)
            {
                int j = 0;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
        }
    }
    else if (1 == layer_type)
    {
        l.coords = 4;
        for (int i = 0; i < total * 2; ++i)
        {
            l.biases[i] = biases_yolov2[i];
        }
    }
    l.layer_type = layer_type;
    l.outputs = l.inputs;
    l.output = ( float* )calloc(batch * l.outputs, sizeof(float));

    return l;
}

void free_darknet_layer(layer l)
{
    if (NULL != l.biases)
    {
        free(l.biases);
        l.biases = NULL;
    }
    if (NULL != l.mask)
    {
        free(l.mask);
        l.mask = NULL;
    }
    if (NULL != l.output)
    {
        free(l.output);
        l.output = NULL;
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n = location / (l.w * l.h);
    int loc = location % (l.w * l.h);
    return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1) + entry * l.w * l.h + loc;
}

void logistic_cpu(float* input, int size)
{
    for (int i = 0; i < size; ++i)
    {
        input[i] = 1.f / (1.f + expf(-input[i]));
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n, b;
    int count = 0;
    for (b = 0; b < l.batch; ++b)
    {
        for (i = 0; i < l.w * l.h; ++i)
        {
            for (n = 0; n < l.n; ++n)
            {
                int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
                if (l.output[obj_index] > thresh)
                    ++count;
            }
        }
    }
    return count;
}

int num_detections(vector<layer> layers_params, float thresh)
{
    int i;
    int s = 0;
    for (i = 0; i < ( int )layers_params.size(); ++i)
    {
        layer l = layers_params[i];
        if (0 == l.layer_type)
            s += yolo_num_detections(l, thresh);
        else if (1 == l.layer_type)
            s += l.w * l.h * l.n;
    }
#if DEBUG_OPTION
    fprintf(stderr, "%s,%d\n", __func__, s);
#endif
    return s;
}

detection* make_network_boxes(vector<layer> layers_params, float thresh, int* num)
{
    layer l = layers_params[0];
    int i;
    int nboxes = num_detections(layers_params, thresh);
    if (num)
        *num = nboxes;
    detection* dets = ( detection* )calloc(nboxes, sizeof(detection));

    for (i = 0; i < nboxes; ++i)
    {
        dets[i].prob = ( float* )calloc(l.classes, sizeof(float));
    }
    return dets;
}

void correct_yolo_boxes(detection* dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if ((( float )netw / w) < (( float )neth / h))
    {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else
    {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i)
    {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / (( float )new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / (( float )new_h / neth);
        b.w *= ( float )netw / new_w;
        b.h *= ( float )neth / new_h;
        if (!relative)
        {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

box get_yolo_box(float* x, float* biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;

    b.x = (i + (x[index + 0 * stride]) * 2 - 0.5) / lw;
    b.y = (j + (x[index + 1 * stride]) * 2 - 0.5) / lh;
    b.w = (4 * pow(x[index + 2 * stride], 2) * biases[2 * n]) / w;
    b.h = (4 * pow(x[index + 3 * stride], 2) * biases[2 * n + 1]) / h;

    return b;
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int* map, int relative,
                        detection* dets)
{
    int i, j, n, b;
    float* predictions = l.output;
    int count = 0;
    for (b = 0; b < l.batch; ++b)
    {
        for (i = 0; i < l.w * l.h; ++i)
        {
            int row = i / l.w;
            int col = i % l.w;
            for (n = 0; n < l.n; ++n)
            {
                int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
                float objectness = predictions[obj_index];
                if (objectness <= thresh)
                    continue;
                int box_index = entry_index(l, b, n * l.w * l.h + i, 0);

                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw,
                                                neth, l.w * l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                for (j = 0; j < l.classes; ++j)
                {
                    int class_index = entry_index(l, b, n * l.w * l.h + i, 4 + 1 + j);
                    float prob = objectness * predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

void correct_region_boxes(detection* dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if ((( float )netw / w) < (( float )neth / h))
    {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else
    {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i)
    {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / (( float )new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / (( float )new_h / neth);
        b.w *= ( float )netw / new_w;
        b.h *= ( float )neth / new_h;
        if (!relative)
        {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

box get_region_box(float* x, float* biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int* map, float tree_thresh,
                           int relative, detection* dets)
{
    int i, j, n;
    float* predictions = l.output;

    for (i = 0; i < l.w * l.h; ++i)
    {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n)
        {
            int index = n * l.w * l.h + i;
            for (j = 0; j < l.classes; ++j)
            {
                dets[index].prob[j] = 0;
            }
            int obj_index = entry_index(l, 0, n * l.w * l.h + i, l.coords);
            int box_index = entry_index(l, 0, n * l.w * l.h + i, 0);
            int mask_index = entry_index(l, 0, n * l.w * l.h + i, 4);
            float scale = predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w * l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if (dets[index].mask)
            {
                for (j = 0; j < l.coords - 4; ++j)
                {
                    dets[index].mask[j] = l.output[mask_index + j * l.w * l.h];
                }
            }
            // int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1);
            if (dets[index].objectness)
            {
                for (j = 0; j < l.classes; ++j)
                {
                    int class_index = entry_index(l, 0, n * l.w * l.h + i, l.coords + 1 + j);
                    float prob = scale * predictions[class_index];
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                }
            }
        }
    }
    correct_region_boxes(dets, l.w * l.h * l.n, w, h, netw, neth, relative);
}

void fill_network_boxes(vector<layer> layers_params, int img_w, int img_h, int net_w, int net_h, float thresh,
                        float hier, int* map, int relative, detection* dets)
{
    int j;
    for (j = 0; j < ( int )layers_params.size(); ++j)
    {
        layer l = layers_params[j];
        if (0 == l.layer_type)
        {
            int count = get_yolo_detections(l, img_w, img_h, net_w, net_h, thresh, map, relative, dets);
            dets += count;
        }
        else
        {
            get_region_detections(l, img_w, img_h, net_w, net_h, thresh, map, hier, relative, dets);
            dets += l.w * l.h * l.n;
        }
    }
}

detection* get_network_boxes(vector<layer> layers_params, int img_w, int img_h, int net_w, int net_h, float thresh,
                             float hier, int* map, int relative, int* num)
{
    // make network boxes
    detection* dets = make_network_boxes(layers_params, thresh, num);

    // fill network boxes
    fill_network_boxes(layers_params, img_w, img_h, net_w, net_h, thresh, hier, map, relative, dets);
    return dets;
}

// release detection memory
void free_detections(detection* dets, int nboxes)
{
    int i;
    for (i = 0; i < nboxes; ++i)
    {
        free(dets[i].prob);
    }
    free(dets);
}

// int nms_comparator(const void* pa, const void* pb)
// {
//     detection a = *( detection* )pa;
//     detection b = *( detection* )pb;
//     float diff = 0;
//     if (b.sort_class >= 0)
//     {
//         diff = a.prob[b.sort_class] - b.prob[b.sort_class];
//     }
//     else
//     {
//         diff = a.objectness - b.objectness;
//     }
//     if (diff < 0)
//         return 1;
//     else if (diff > 0)
//         return -1;
//     return 0;
// }

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0)
        return 0;
    float area = w * h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b) / box_union(a, b);
}

void do_nms_sort(detection* dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i)
    {
        if (dets[i].objectness == 0)
        {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k)
    {
        for (i = 0; i < total; ++i)
        {
            dets[i].sort_class = k;
        }
        // qsort(dets, total, sizeof(detection), nms_comparator);
        for (i = 0; i < total; ++i)
        {
            if (dets[i].prob[k] == 0)
                continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j)
            {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh)
                {
                    dets[j].prob[k] = 0;
				}
            }
        }  
    }
}


void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-d video_device] [-r repeat_count] [-t thread_count]\n");
}

int check_file_exist(const char* file_name)
{
    FILE* fp = fopen(file_name, "r");
    if (!fp)
    {
        fprintf(stderr, "Input file not existed: %s\n", file_name);
        return 0;
    }
    fclose(fp);
    return 1;
}


static cv::Scalar obj_id_to_color(int obj_id) {
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	int const offset = obj_id * 123457 % 6;
	int const color_scale = 150 + (obj_id * 123457) % 100;
	cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
	color *= color_scale;
	return color;
}



void postpress_graph(cv::Mat& frame, graph_t graph, int output_node_num,int net_w, int net_h,int numBBoxes,int total_numAnchors,int layer_type,char *window_name)
{
//	cv::Mat frame = sample;

	vector<layer> layers_params;
    //coco datasheet
    const char *coco_names[80] = {"person","bicycle","car","motorbike","aeroplane","bus",
                                "train","truck","boat","traffic light","fire hydrant","stop sign",
                                "parking meter","bench","bird","cat","dog","horse","sheep","cow",
                                "elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
                                "tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
                                "baseball bat","baseball glove","skateboard","surfboard","tennis racket",
                                "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
                                "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
                                "chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
                                "laptop","mouse","remote","keyboard","cell phone","microwave","oven",
                                "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};
	layers_params.clear();

	for (int i = 0; i < output_node_num; ++i)
	{
		tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);    //"detection_out"
		int out_dim[4];
		get_tensor_shape(out_tensor, out_dim, 4);
		layer l_params;
		int out_w = out_dim[3];
		int out_h = out_dim[2];
		l_params = make_darknet_layer(1, out_w, out_h, net_w, net_h, numBBoxes, total_numAnchors, classes, layer_type);
		layers_params.push_back(l_params);

		/* dequant output data */
		float output_scale = 0.f;
		int output_zero_point = 0;
		get_tensor_quant_param(out_tensor, &output_scale, &output_zero_point, 1);
		int count = get_tensor_buffer_size(out_tensor) / sizeof(uint8_t);
		uint8_t* data_uint8 = ( uint8_t* )get_tensor_buffer(out_tensor);
		float* data_fp32 = ( float* )malloc(sizeof(float) * count);
		for (int c = 0; c < count; c++)
		{
			data_fp32[c] = (( float )data_uint8[c] - ( float )output_zero_point) * output_scale;
		}
        /* copy data to darknet layer*/
        memcpy(( void* )l_params.output, ( void* )data_fp32, sizeof(float) * l_params.inputs * l_params.batch);

		free(data_fp32);
	}

	int nboxes = 0;
	// get network boxes
	detection* dets =
		get_network_boxes(layers_params, frame.cols, frame.rows, net_w, net_h, thresh, hier_thresh, 0, relative, &nboxes);

	if (nms != 0)
	{
		do_nms_sort(dets, nboxes, classes, nms);
	}

	int i, j;
	for (i = 0; i < nboxes; ++i)
	{
		int cls = -1;
		char cvTest[64];
		for (j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j] > 0.5)
			{
				if (cls < 0)
				{
					cls = j;
				}
#if DEBUG_OPTION
				fprintf(stderr, "%d: %.0f%%\n", cls, dets[i].prob[j] * 100);
#endif
				sprintf(cvTest,"%s",coco_names[cls]);
			}

		}
		if (cls >= 0)
		{
			box b = dets[i].bbox;
			int left = (b.x - b.w / 2.) * frame.cols;
			int top = (b.y - b.h / 2.) * frame.rows;
			if (top < 30) {
				top = 30;
				left +=10;
			}

#if DEBUG_OPTION
			fprintf(stderr, "left = %d,top = %d\n", left, top);
#endif
			cv::Rect rect(left, top, b.w*frame.cols, b.h*frame.rows);

			int baseline;
			cv::Size text_size = cv::getTextSize(cvTest, cv::FONT_HERSHEY_COMPLEX,0.5,1,&baseline);

			cv::Rect rect1(left, top-20, text_size.width+10, 20);
			cv::rectangle(frame,rect,obj_id_to_color(cls),1,8,0);
			cv::rectangle(frame,rect1,obj_id_to_color(cls),-1);
			cv::putText(frame,cvTest,cvPoint(left+5,top-5),cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,0,0),1);
		}


		if (dets[i].mask)
			free(dets[i].mask);
		if (dets[i].prob)
			free(dets[i].prob);
	}
	free(dets);

	for (int i = 0; i < (int)layers_params.size(); i++)
	{
		layer l = layers_params[i];
		if (l.output)
			free(l.output);
		if (l.biases)
			free(l.biases);
		if (l.mask)
			free(l.mask);
	}

	cv::imshow(window_name, frame);
	cv::waitKey(1); 
//	frame.release();
}

void postpress_graph_image_wrapper(void* data_pointer, int height, int width, float* array, graph_t graph, int output_node_num,int net_w, int net_h,int numBBoxes,int total_numAnchors,int layer_type)
{
	cv::Mat frame = cv::Mat(height, width, CV_8UC3, (uchar*)data_pointer);
    const char *coco_names[80] = {"person","bicycle","car","motorbike","aeroplane","bus",
                                "train","truck","boat","traffic light","fire hydrant","stop sign",
                                "parking meter","bench","bird","cat","dog","horse","sheep","cow",
                                "elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
                                "tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
                                "baseball bat","baseball glove","skateboard","surfboard","tennis racket",
                                "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
                                "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
                                "chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
                                "laptop","mouse","remote","keyboard","cell phone","microwave","oven",
                                "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};
	std::vector<layer> layers_params;
	layers_params.clear(); 

	for (int i = 0; i < output_node_num; ++i)
	{
		tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);    //"detection_out"
		int out_dim[4];
		get_tensor_shape(out_tensor, out_dim, 4);
		layer l_params;
		int out_w = out_dim[3];
		int out_h = out_dim[2];
		l_params = make_darknet_layer(1, out_w, out_h, net_w, net_h, numBBoxes, total_numAnchors, classes, layer_type);
		layers_params.push_back(l_params);

		/* dequant output data */
		float output_scale = 0.f;
		int output_zero_point = 0;
		get_tensor_quant_param(out_tensor, &output_scale, &output_zero_point, 1);
		int count = get_tensor_buffer_size(out_tensor) / sizeof(uint8_t);
		uint8_t* data_uint8 = ( uint8_t* )get_tensor_buffer(out_tensor);
		float* data_fp32 = ( float* )malloc(sizeof(float) * count);
		for (int c = 0; c < count; c++)
		{
			data_fp32[c] = (( float )data_uint8[c] - ( float )output_zero_point) * output_scale;
		}
        /* copy data to darknet layer */
        memcpy(( void* )l_params.output, ( void* )data_fp32, sizeof(float) * l_params.inputs * l_params.batch);

		free(data_fp32);
	}

	int nboxes = 0;
	// get network boxes
	detection* dets =
		get_network_boxes(layers_params, frame.cols, frame.rows, net_w, net_h, thresh, hier_thresh, 0, relative, &nboxes);

	if (nms != 0)
	{
		do_nms_sort(dets, nboxes, classes, nms);
	}
    int counter = 0;
	int i, j;
	for (i = 0; i < nboxes; ++i)
	{
		int cls = -1;
		char cvTest[64];
		for (j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j] > thresh)
			{
				if (cls < 0)
				{
					cls = j;
				}
				fprintf(stderr, "%d: %.0f%%\n", cls, dets[i].prob[j] * 100);
                array[counter*6 + 5] = (float) (dets[i].prob[j] * 100);
//				sprintf(cvTest,"%s:%.0f%%",coco_names[cls],dets[i].prob[j]*100);
				sprintf(cvTest,"%s",coco_names[cls]);
			}

		}
		if (cls >= 0)
		{
			box b = dets[i].bbox;
			int left = (b.x - b.w / 2.)  * frame.cols;
			int top = (b.y - b.h / 2.) * frame.rows;
			if (top < 30) {
				top = 30;
				left +=10;
			}
 
			fprintf(stderr, "left = %d,top = %d, width = %.1f, height = %.1f\n", left, top, b.w, b.h);
            array[counter*6] = (float) left;
            array[counter*6 + 1] = (float) top;
            array[counter*6 + 2] = (float) (b.w*frame.cols);
            array[counter*6 + 3] = (float) (b.h*frame.rows);
            array[counter*6 + 4] = (float) cls;
            counter++;
			cv::Rect rect(left, top, b.w*frame.cols, b.h*frame.rows);

			int baseline;
			cv::Size text_size = cv::getTextSize(cvTest, cv::FONT_HERSHEY_COMPLEX,0.5,1,&baseline);
			cv::Rect rect1(left, top-20, text_size.width+10, 20);
			cv::rectangle(frame,rect,obj_id_to_color(cls),2,20,0);
			cv::rectangle(frame,rect1,obj_id_to_color(cls),-1);
			cv::putText(frame,cvTest,cvPoint(left+5,top-5),cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,0,0),1);
		}

        
		if (dets[i].mask)
			free(dets[i].mask);
		if (dets[i].prob)
			free(dets[i].prob);
	}
    array[counter*6 + 5] = 0;
	free(dets);

	for (int i = 0; i < (int)layers_params.size(); i++)
	{
		layer l = layers_params[i];
		if (l.output)
			free(l.output);
		if (l.biases)
			free(l.biases);
		if (l.mask)
			free(l.mask);
	}
    // cv::imwrite("detection.jpg", frame);
	// cv::namedWindow("MyWindow", 0x00000001); // CV_WINDOW_AUTOSIZE
	// cv::imshow("MyWindow", frame);
	// cv::waitKey(0);
}

void postpress_graph_image(const cv::Mat& sample, graph_t graph, int output_node_num,int net_w, int net_h,int numBBoxes,int total_numAnchors,int layer_type)
{
	cv::Mat frame = sample;
    const char *coco_names[80] = {"person","bicycle","car","motorbike","aeroplane","bus",
                                "train","truck","boat","traffic light","fire hydrant","stop sign",
                                "parking meter","bench","bird","cat","dog","horse","sheep","cow",
                                "elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
                                "tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
                                "baseball bat","baseball glove","skateboard","surfboard","tennis racket",
                                "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
                                "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
                                "chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
                                "laptop","mouse","remote","keyboard","cell phone","microwave","oven",
                                "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};
	std::vector<layer> layers_params;
	layers_params.clear();

	for (int i = 0; i < output_node_num; ++i)
	{
		tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);    //"detection_out"
		int out_dim[4];
		get_tensor_shape(out_tensor, out_dim, 4);
		layer l_params;
		int out_w = out_dim[3];
		int out_h = out_dim[2];
		l_params = make_darknet_layer(1, out_w, out_h, net_w, net_h, numBBoxes, total_numAnchors, classes, layer_type);
		layers_params.push_back(l_params);

		/* dequant output data */
		float output_scale = 0.f;
		int output_zero_point = 0;
		get_tensor_quant_param(out_tensor, &output_scale, &output_zero_point, 1);
		int count = get_tensor_buffer_size(out_tensor) / sizeof(uint8_t);
		uint8_t* data_uint8 = ( uint8_t* )get_tensor_buffer(out_tensor);
		float* data_fp32 = ( float* )malloc(sizeof(float) * count);
		for (int c = 0; c < count; c++)
		{
			data_fp32[c] = (( float )data_uint8[c] - ( float )output_zero_point) * output_scale;
		}
        /* copy data to darknet layer */
        memcpy(( void* )l_params.output, ( void* )data_fp32, sizeof(float) * l_params.inputs * l_params.batch);

		free(data_fp32);
	}

	int nboxes = 0;
	// get network boxes
	detection* dets =
		get_network_boxes(layers_params, frame.cols, frame.rows, net_w, net_h, thresh, hier_thresh, 0, relative, &nboxes);

	if (nms != 0)
	{
		do_nms_sort(dets, nboxes, classes, nms);
	}

	int i, j;
	for (i = 0; i < nboxes; ++i)
	{
		int cls = -1;
		char cvTest[64];
		for (j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j] > thresh)
			{
				if (cls < 0)
				{
					cls = j;
				}
				fprintf(stderr, "%d: %.0f%%\n", cls, dets[i].prob[j] * 100);
//				sprintf(cvTest,"%s:%.0f%%",coco_names[cls],dets[i].prob[j]*100);
				sprintf(cvTest,"%s",coco_names[cls]);
			}

		}
		if (cls >= 0)
		{
			box b = dets[i].bbox;
			int left = (b.x - b.w / 2.)  * frame.cols;
			int top = (b.y - b.h / 2.) * frame.rows;
			if (top < 30) {
				top = 30;
				left +=10;
			}

			fprintf(stderr, "left = %d,top = %d, width = %.1f, height = %.1f\n", left, top, b.w, b.h);
			cv::Rect rect(left, top, b.w*frame.cols, b.h*frame.rows);

			int baseline;
			cv::Size text_size = cv::getTextSize(cvTest, cv::FONT_HERSHEY_COMPLEX,0.5,1,&baseline);
			cv::Rect rect1(left, top-20, text_size.width+10, 20);
			cv::rectangle(frame,rect,obj_id_to_color(cls),2,20,0);
			cv::rectangle(frame,rect1,obj_id_to_color(cls),-1);
			cv::putText(frame,cvTest,cvPoint(left+5,top-5),cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,0,0),1);
		}


		if (dets[i].mask)
			free(dets[i].mask);
		if (dets[i].prob)
			free(dets[i].prob);
	}

	free(dets);

	for (int i = 0; i < (int)layers_params.size(); i++)
	{
		layer l = layers_params[i];
		if (l.output)
			free(l.output);
		if (l.biases)
			free(l.biases);
		if (l.mask)
			free(l.mask);
	}
    cv::imwrite("detection.jpg", frame);
	// cv::namedWindow("MyWindow", 0x00000001); // CV_WINDOW_AUTOSIZE
	// cv::imshow("MyWindow", frame);
	// cv::waitKey(0);
}

void thread_camera()
{
    pthread_mutex_t mutex4q;
	int layer_type = 0;
	int numBBoxes = 3;
	int total_numAnchors = 9;
	int net_w = MODEL_COLS;
	int net_h = MODEL_ROWS;

    char* model_file = nullptr;
	char *window_name = (char *)"CameraWindow";
    char *video_device = NULL;
    
	string str = video_device;  
	
	cv::VideoCapture cap(str);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WEIGHT);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
	
	if (!cap.isOpened()) {
		cout << "capture device failed to open!" << endl;
		cap.release();
	}

	setpriority(PRIO_PROCESS, pthread_self(), -15);

	cv::namedWindow(window_name);

 
	/* inital tengine */
	init_tengine();
	fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());


	/* set inference context with npu */
	context_t timvx_context = create_context("timvx", 1);

	int rtt = set_context_device(timvx_context, VXDEVICE, nullptr, 0);
	if (0 > rtt)
	{
		fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
		exit(-1);
	}

	/* create graph, load tengine model xxx.tmfile */
	graph_t graph = create_graph(timvx_context, "tengine", model_file);
	if (graph == nullptr)
	{
		fprintf(stderr, "Create graph failed.\n");
		fprintf(stderr, "errno: %d \n", get_tengine_errno());
		exit(-1);
	}


	/* set the input shape to initial the graph, and prerun graph to infer shape */
	int img_size = net_h * net_w * 3;
	int dims[] = {1, 3, net_h, net_w};    // nchw

	std::vector<uint8_t> input_data(img_size);

	tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
	if (input_tensor == nullptr)
	{
		fprintf(stderr, "Get input tensor failed\n");
		exit(-1);
	}

	if (set_tensor_shape(input_tensor, dims, 4) < 0)
	{
		fprintf(stderr, "Set input tensor shape failed\n");
		exit(-1);
	}

	if (prerun_graph(graph) < 0)
	{
		fprintf(stderr, "Prerun graph failed\n");
		exit(-1);
	}

	float input_scale = 0.f;
	int input_zero_point = 0;


	get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);

	if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0)
	{
		fprintf(stderr, "Set input tensor buffer failed\n");
		exit(-1);
	}

	int output_node_num = get_graph_output_node_number(graph);
    int counter = 1;
    int frames = 30;
    double start;
    int repeat_count = DEFAULT_REPEAT_COUNT;

	while(1){

        if (counter==1){
            start = get_current_time();
        }

		cv::Mat frame(CAMERA_HEIGHT,CAMERA_WEIGHT,CV_8UC3);
		pthread_mutex_lock(&mutex4q);
		if (!cap.read(frame)) {
			pthread_mutex_unlock(&mutex4q);
			cout<<"Capture read error"<<std::endl;
			break;
		}

		pthread_mutex_unlock(&mutex4q);

		get_input_data_cv(frame,input_data.data(),net_w,net_h,input_scale, input_zero_point,0);


		/* run graph */	
		for (int i = 0; i < repeat_count; i++)
		{	
			if (run_graph(graph, 1) < 0)
			{
				fprintf(stderr, "Run graph failed\n");
				exit(-1);
			}	
		}

		/* process the detection result */
		int output_node_num = get_graph_output_node_number(graph);
	
		postpress_graph(frame,graph,output_node_num,net_w,net_h,numBBoxes,total_numAnchors,layer_type,window_name);

		frame.release();

        if (counter%frames==0) {
            double end = get_current_time();
            printf("FPS: %g\n", frames*1000/(end-start));
            counter = 0;
        } 

        counter++;      
	}
	/* release tengine */
	for (int i = 0; i < output_node_num; ++i)
	{
		tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);
		release_graph_tensor(out_tensor);
	}


	release_graph_tensor(input_tensor);
	postrun_graph(graph);
	destroy_graph(graph);
	release_tengine();
	// return 0;
}
int set_graph(int net_h, int net_w, graph_t graph){
    int img_size = net_h * net_w * 3;
    int dims[] = {1, 3, net_h, net_w};    // nchw
 
    vector<uint8_t> input_data(img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (prerun_graph(graph) < 0)
    {
        fprintf(stderr, "Prerun graph failed\n");
        return -1;
    }
    return 0;
}
int set_image_wrapper(void* data_pointer, int height, int width, tensor_t input_tensor, int net_h, int net_w){
    float input_scale = 0.f;
    int input_zero_point = 0;
    int img_size = net_h * net_w * 3;
    vector<uint8_t> input_data(img_size);
	// cv::Mat frame = cv::imread(image_file,cv::IMREAD_COLOR);
    cv::Mat frame = cv::Mat(height, width, CV_8UC3, (uchar*)data_pointer);
	if (frame.empty())
	{
		fprintf(stderr, "Empty buffer\n");
		return -1;
	}
    get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
    if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }
	get_input_data_cv(frame,input_data.data(),net_w,net_h,input_scale, input_zero_point, 0);
    return 0;
}

int set_image(const char* image_file, tensor_t input_tensor, int net_h, int net_w){
    float input_scale = 0.f;
    int input_zero_point = 0;
    int img_size = net_h * net_w * 3;
    vector<uint8_t> input_data(img_size);
	cv::Mat frame = cv::imread(image_file,cv::IMREAD_COLOR);
	if (frame.empty())
	{
		fprintf(stderr, "cv::imread %s failed\n", image_file);
		return -1;
	}

    get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
    if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

	get_input_data_cv(frame,input_data.data(),net_w,net_h,input_scale, input_zero_point, 0);
    return 0;
}

int inference(const char* image_file, const char* model_file,
            context_t timvx_context, int rtt){
    int repeat_count = 1; 
    int num_thread = 4;

    int layer_type = 0;
    int numBBoxes = 3;
    int total_numAnchors = 9;
    int net_w = 352;
    int net_h = 352;
    printf("Image: %s\n", image_file);

    /* check files */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (nullptr == image_file)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    /* inital tengine */
    // init_tengine();
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

	/* set inference context with npu */
	// context_t timvx_context = create_context("timvx", 1);

	// int rtt  = set_context_device(timvx_context, VXDEVICE, nullptr, 0);
	if (0 > rtt)
	{
		fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
		exit(-1);
	}

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(timvx_context, "tengine", model_file);
    if (graph == nullptr)
    {
        fprintf(stderr, "Create graph failed.\n");
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = net_h * net_w * 3;
    int dims[] = {1, 3, net_h, net_w};    // nchw

    vector<uint8_t> input_data(img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (prerun_graph(graph) < 0)
    {
        fprintf(stderr, "Prerun graph failed\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    float input_scale = 0.f;
    int input_zero_point = 0;
	cv::Mat frame = cv::imread(image_file,cv::IMREAD_COLOR);
	if (frame.empty())
	{
		fprintf(stderr, "cv::imread %s failed\n", image_file);
		return -1;
	}

    get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1); 
    if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

	get_input_data_cv(frame,input_data.data(),net_w,net_h,input_scale, input_zero_point, 0);
    /* run graph */
    double min_time = __DBL_MAX__;
    double max_time = -__DBL_MAX__;
    double total_time = 0.;
    for (int i = 0; i < repeat_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        min_time = min(min_time, cur);
        max_time = max(max_time, cur);
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
            num_thread, total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* process the detection result */
    int output_node_num = get_graph_output_node_number(graph);

	postpress_graph_image(frame,graph,output_node_num,net_w,net_h,numBBoxes,total_numAnchors,layer_type);

	/* release tengine */
    for (int i = 0; i < output_node_num; ++i)
    {
        tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);
        release_graph_tensor(out_tensor);
    } 


    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    // release_tengine();

    return 0;
}

