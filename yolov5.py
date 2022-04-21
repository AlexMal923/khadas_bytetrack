from ctypes import *
import cv2
import time 
import copy
import numpy as np
import threading
libc = cdll.LoadLibrary("./yolov5_lib.so")
libc.set_image_wrapper.argtypes = [c_void_p, c_int, c_int, c_void_p, c_int, c_int]
libc.postpress_graph_image_wrapper.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p,
                                                c_int , c_int, c_int, c_int, c_int, c_int]
class YOLOV5(threading.Thread):
    def __init__(self, model):
        threading.Thread.__init__(self)
        self.context = libc.create_context("timvx".encode('utf-8'), 1)
        libc.init_tengine()
        rtt = libc.set_context_device(self.context, "TIMVX".encode('utf-8'), None, 0)
        model_file = model.encode('utf-8')
        self.graph = libc.create_graph(self.context, "tengine".encode('utf-8'), model_file)
        libc.set_graph(352, 352, self.graph)
        self.input_tensor = libc.get_graph_input_tensor(self.graph, 0, 0)
        self.output_node_num = libc.get_graph_output_node_number(self.graph)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.dets = np.zeros([30,6], dtype = np.float32)
        self.last_frame = None
        self.last_dets = None
        print("Initialised")
    def inference(self):
        while True:  
            ret, self.frame = self.cap.read()
            height, width, _ = self.frame.shape
            self.frame = self.frame[:,:,::-1]
            libc.set_image_wrapper(self.frame.ctypes.data, height, width, self.input_tensor, 352, 352)
            libc.run_graph(self.graph, 1)
            libc.postpress_graph_image_wrapper(self.frame.ctypes.data, height, width, self.dets.ctypes.data, self.graph,self.output_node_num, 352,352,3,9,0)
            self.last_frame = copy.deepcopy(self.frame[::,::,::-1])
            self.last_dets = copy.deepcopy(self.dets)
            cv2.imshow('frame', self.last_frame)
            key = cv2.waitKey(1)
            if (key == 27):
                break
                
    def run(self):
        self.inference()
        
if __name__ = "__main__":
  yolov5 = YOLOV5("yolov5m_leaky_352_0mean_uint8.tmfile")
  yolov5.start()
  yolov5.setDaemon(True)
  while True:
      pass
