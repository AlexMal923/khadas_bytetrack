from ctypes import *
from multiprocessing import shared_memory
import cv2
import time 
import copy
import numpy as np
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
BUF_SZ = 10
NUM_PROC = 2
PROC = 1
CLASSES = 1
NUM_DETS = 300
#MODEL = "yolov5m_bubbles.tmfile"
MODEL = "bubbles_3.tmfile"
#MODEL = "yolov5m_leaky_352_0mean_uint8.tmfile"
libc = cdll.LoadLibrary("./yolov5_lib.so")
libc.set_image_wrapper.argtypes = [c_void_p, c_int, c_int, c_void_p, c_int, c_int]
libc.postpress_graph_image_wrapper.argtypes = [c_int, c_int, c_void_p, c_void_p,
												c_int , c_int, c_int, c_int, c_int, c_float]
								
class YOLOV5(threading.Thread):
	def __init__(self, model, daemon = True):
		threading.Thread.__init__(self)
		self.daemon = daemon	
		self.img_sz = 352
		self.context = libc.create_context("timvx".encode('utf-8'), 1)
		libc.init_tengine()
		rtt = libc.set_context_device(self.context, "TIMVX".encode('utf-8'), None, 0)
		model_file = model.encode('utf-8')
		self.graph = libc.create_graph(self.context, "tengine".encode('utf-8'), model_file)
		libc.set_graph(352 , self.img_sz , self.graph)
		self.input_tensor = libc.get_graph_input_tensor(self.graph, 0, 0)
		self.output_node_num = libc.get_graph_output_node_number(self.graph)
		self.dets = np.zeros([NUM_DETS,6], dtype = np.float32)  
		self.classes = CLASSES
		# self.last_frame = None
		self.last_dets = None
		self.current = -1
		# self.draw = True
		print("Initialised")
		self.ex_pre = shared_memory.SharedMemory(name="pre") #preprocessed images
		self.ex_frm= shared_memory.SharedMemory(name="frame") # raw image
		self.ex_counter = shared_memory.SharedMemory(name="counter") # number of images read from camera
		self.ex_status = shared_memory.SharedMemory(name="status") # not inferenced images
		self.ex_read = mp.shared_memory.SharedMemory(name = "read") # number of element to read from
		self.ex_dets = mp.shared_memory.SharedMemory(name = "dets") 
		self.pre = np.ndarray([BUF_SZ, 3, 352, 352], dtype=np.uint8, buffer=self.ex_pre.buf)
		self.counter = np.ndarray([1], dtype=np.int64, buffer=self.ex_counter.buf)
		self.frm =  np.ndarray([BUF_SZ, 480, 640, 3], dtype=np.uint8, buffer=self.ex_frm.buf)
		self.status = np.ndarray([10], dtype=np.uint8, buffer=self.ex_status.buf)
		self.read = np.ndarray([NUM_PROC], dtype=np.int64, buffer=self.ex_read.buf)
		self.dets_buf = np.ndarray([BUF_SZ, NUM_DETS, 6], dtype=np.float32, buffer=self.ex_dets.buf)

	def inference(self):
		if self.status[(self.counter[0]-1)%BUF_SZ] and (self.counter[0]-1)%NUM_PROC == PROC:
			self.current = (self.counter[0]-1)
			self.status[self.current%BUF_SZ] = 0 # change status to inferenced
			start_main = time.time()  
			self.frame = self.pre[self.current%BUF_SZ]
			libc.set_image_wrapper(self.frame.ctypes.data, 352, 352, self.input_tensor, self.img_sz, self.img_sz )
			start = time.time()
			libc.run_graph(self.graph, 1)
			print("Graph ran for ", time.time() - start)
			# print("Fps: ", 1/(time.time() - start_main))
			# print("Run graph fps: ", 1/(time.time() - start))
			start = time.time()
			libc.postpress_graph_image_wrapper(480, 640, self.dets.ctypes.data, 
												self.graph,self.output_node_num, 352 ,self.img_sz, self.classes, NUM_DETS, 0.2)
			# self.last_frame = copy.deepcopy(self.frame)
			self.last_dets = copy.deepcopy(self.dets)
			print("Frame number: ", self.current)
			print("Full fps: ", 1/(time.time() - start_main))
			# print("Dets: ", self.last_dets[0])
			self.read[PROC] = self.current #last inferenced image from 0-9
			self.dets_buf[self.current%BUF_SZ][:] = self.dets[:]
			# print("Write to ", self.current, time.time())
			# print("Dets buf: ", self.dets_buf)

	def run(self):
		while True:
			self.inference()
			

if __name__ == "__main__":
	yolov5 = YOLOV5(MODEL, daemon = False)	
	yolov5.start()