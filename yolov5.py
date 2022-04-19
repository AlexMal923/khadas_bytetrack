import ctypes 
import cv2
import time
import copy
import threading
import multiprocessing as mp
import numpy as np
libc = ctypes.cdll.LoadLibrary("./yolov5_lib.so")
from multiprocessing import Process

class YOLOV5(threading.Thread):
    def __init__(self, model):
        threading.Thread.__init__(self)
        self.init = False
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
        self.imsize = 352
        self.context = libc.create_context("timvx".encode('utf-8'), 1)
        if libc.init_tengine():
            print("Tengine not initialised")
        print("Setting context device")
        rtt = libc.set_context_device(self.context, "TIMVX".encode('utf-8'), None, 0)
        print("Rtt:", rtt)
        print("Creating graph")
        self.model_file = model
        self.graph = libc.create_graph(self.context, "tengine".encode('utf-8'), self.model_file.encode('utf-8'))
        print("Setting graph: ", self.graph)
        libc.set_graph(self.imsize, self.imsize, self.graph)
        print("Getting input tensor")
        self.input_tensor = libc.get_graph_input_tensor(self.graph, 0, 0)
        print("Getting output node num")
        self.output_node_num = libc.get_graph_output_node_number(self.graph)
        print("Initialised!")
        start = time.time()
        libc.set_image_wrapper.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        libc.postpress_graph_image_wrapper.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_int , ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        print("Set function for: ", time.time() - start)
        self.frame = None
        self.dets = np.zeros([30,6], dtype = np.float32)
        self.init = True
        self.inferenced = False

    def inference(self):
        start = time.time()
        self.ret, self.frame = self.cap.read()
        # if not self.ret:
        #     return
        print("Ret: ", self.ret)
        # self.frame = cv2.resize(self.frame, (352,264))
        # print("7")
        self.frame = self.frame[:,:, ::-1]
        # # height, width, _ = self.frame.shape
        # print(self.frame.ctypes.data, 288, 352, self.input_tensor, self.imsize, self.imsize)
        
        libc.set_image_wrapper(self.frame.ctypes.data, 288, 352, self.input_tensor, self.imsize, self.imsize)
        print("Set images for  ", time.time() - start)
        start = time.time()
        libc.run_graph(self.graph, 1)
        print("Graph ran for ", time.time() - start)
        start = time.time()
        self.last_frame = copy.deepcopy(self.frame[:,:, ::-1])
        print("Copied picture for ", time.time() - start)
        self.inferenced = True

    def post(self):
        while self.inferenced:
            start = time.time()            
            libc.postpress_graph_image_wrapper(self.last_frame.ctypes.data, 288, 352, self.dets.ctypes.data, self.graph,self.output_node_num, self.imsize, self.imsize,3,9,0)
            print("Postprocess ran for ", time.time() - start)
            self.inferenced = False
            cv2.imshow('frame', self.last_frame)
            key = cv2.waitKey(1)
            if (key == 27):
                break

    def run(self):
        while True:
            if self.init:
                start = time.time()
                # time.sleep(1)
                self.inference()
                self.post()
                print("Fps:", 1/(time.time() - start))
        # self.post()

    def __del__(self):
        self.cap.release()
        libc.release_graph_tensor(self.input_tensor)
        libc.postrun_graph(self.graph)
        libc.destroy_graph(self.graph)
        libc.release_tengine()
        print("Released everything and deleted yolov5")

class InferThread(threading.Thread):
    def __init__(self, yolov5):
        threading.Thread.__init__(self)
        self.yolov5 = yolov5

    def run(self):
        print("Running!")
        print(yolov5.dets)
        yolov5.inference()
        yolov5.post()

class PostThread(threading.Thread):
    def __init__(self, yolov5):
        threading.Thread.__init__(self)
        self.yolov5 = yolov5

    def run(self):
       
        self.yolov5.post()
    

# def inf():
#     while True:
#         # print("inf")
#         yolov5.inference()

# def post():
#     while True:
#         # print("post")
#         yolov5.post()

# def runInParallel(*fns):
#     proc = []
#     for fn in fns:
#         p = mp.Process(target=fn)
#         p.start()
#         proc.append(p)
#     for p in proc:
#         p.join()


if __name__ == "__main__":
    # global yolov5
    # libc.release_tengine()
    
    
    # inf = InferThread(yolov5)
    # 
    # # time.sleep(15)
    #
    try:
        yolov5 = YOLOV5("yolov5m_leaky_352_0mean_uint8.tmfile")
        yolov5.start()  
        yolov5.setDaemon(True)  
        while True:
            pass
    except KeyboardInterrupt:
        del yolov5

    # all_processes =[yolov5.inference, yolov5.post]
    # 
    # try:
    #     yolov5 = YOLOV5("yolov5m_leaky_352_0mean_uint8.tmfile")
    #     while True:
    #         yolov5.inference()
    #         yolov5.post()
    # except KeyboardInterrupt:
    #     del yolov5
    # try:
    #     infer = threading.Thread(target = yolov5.inference, daemon = True)
    #     infer.start()
    #     while True:
    #         infer.join()
    # except:
    #     del yolov5
    # while True:
    #     start = time.time()
    #     yolov5.inference()
    #     yolov5.post()
    #     # cv2.imshow('frame', yolov5.frame)
    #     # key = cv2.waitKey(1)
    #     # if (key == 27):
    #     #     break
    #     print("Fps: ", 1/(time.time()-start))

