import cv2
import numpy as np
from multiprocessing import shared_memory,Process
import time
from yolov5 import run
from pre import cam

BUF_SZ = 10
NUM_PROC = 2
NUM_DETS = 300
SOURCE = 0
MODEL = "yolov5m_leaky_352_0mean_uint8.tmfile"

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

class Khadas():
    def __init__(self):
        self.cam = Process(target=cam, args = (SOURCE,))
        self.cam.start()
        time.sleep(1)    
        self.ex_stop = shared_memory.SharedMemory(name = "stop") 
        self.stop =  np.ndarray([1], dtype=np.uint8, buffer=self.ex_stop.buf)  
        self.last_model = MODEL
        self.upload_models(MODEL)
        self.ex_frm = shared_memory.SharedMemory(name="frame")
        self.ex_read = shared_memory.SharedMemory(name="read")
        self.ex_dets = shared_memory.SharedMemory(name = "dets")
        self.ex_status = shared_memory.SharedMemory(name="status") # not inferenced images = 1
        self.frm =  np.ndarray([BUF_SZ, 480, 640, 3], dtype=np.uint8, buffer=self.ex_frm.buf)
        self.read = np.ndarray([NUM_PROC], dtype=np.int64, buffer=self.ex_read.buf)		 
        self.dets_buf = np.ndarray([BUF_SZ, NUM_DETS, 6], dtype=np.float32, buffer=self.ex_dets.buf)
        self.status = np.ndarray([BUF_SZ], dtype=np.uint8, buffer=self.ex_status.buf)
        self.counter = [-1]*NUM_PROC
        self.begin = time.time()
        self.frame_counter = 0
        self.proc = 0
        self.last = np.amin(self.read)
        self.temp = 0
        self.fps = 0
        self.max_fps = 0
        self.thresh = 10
        self.frame = None
        self.dets = None
        self.tracker = None


    def upload_models(self, model, change = False):
        self.stop[0] = 1
        if change:
            self.m1.kill()
            self.m2.kill()
        self.m1 = Process(target=run, args = (0, model))
        self.m1.start()
        self.m2 = Process(target=run, args = (1, model))
        self.m2.start()
        if change:
            time.sleep(5)  # wait so that model can start     
            self.m1.join(timeout=0)
            self.m2.join(timeout=0)
            if not self.m1.is_alive() or not self.m2.is_alive():
                self.stop[0] = 1
                self.m1.kill()
                self.m2.kill()
                self.m1 = Process(target=run, args = (0, self.last_model))
                self.m1.start()
                self.m2 = Process(target=run, args = (1, self.last_model))
                self.m2.start()
                print("Corrupted model!")
            else:
                self.last_model = model  


    def show(self):
        if np.amin(self.read) - self.last:
            diff = np.amin(self.read) - self.last
            if diff > BUF_SZ:
                self.last = np.amin(self.read)
                return
            print("Difference:", diff)
            print(np.amin(self.read), self.last)
            for i in range(1,(diff+1)%BUF_SZ):
                if not self.status[(self.last+i)%BUF_SZ]:# если инференс был на одной из картинок, которые не успели показать
                    self.temp = self.last+i
                    print("Reading from: ", self.temp)
                    self.frame_counter+=1
                    frame = self.frm[self.temp%BUF_SZ]
                    self.dets = self.dets_buf[self.temp%BUF_SZ]
                    self.frame = self.post(frame, self.dets)
                    if not self.frame_counter % 30:
                        self.fps = 30/(time.time() - self.begin)
                        if self.fps > self.max_fps:
                            self.max_fps = self.fps
                        self.begin = time.time()
                    cv2.imshow('frame', self.frame)
                    key = cv2.waitKey(1)
                else: print("Not inferenced:", self.last+i)

            if self.temp:
                self.last = self.temp
                self.temp = 0		
            print("Max FPS %.2f, Current Fps: %.2f"%(self.max_fps, self.fps))

    def post(self, frame, dets):
        for det in dets:
            if det[5] == 0:
                return frame
            if det[5] < self.thresh:
                continue
            b = 100 + (25 * det[4]) % 156
            g = 100 + (80 + 40 * det[4]) % 156
            r = 100 + (120 + 60 * det[4]) % 156
            color = (b, g, r)
            start = (int(det[0]), int(det[1])) 
            end =(int(det[0]+det[2]), int(det[1] + det[3])) 
            cv2.rectangle(frame,start, end, color, 2)
            text = "%.0f "%(det[5]) + names[int(det[4])]
            cv2.putText(frame,text,
                        (start[0]+5, start[1]-5),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        color, 1)
            # cv2.imwrite(f'test_{time.time()}.jpg', frame)
        return frame
    
    def tracking_post(self, frame, dets):
        for det in dets:
            if not det.size:
                continue
            b = 100 + (25 * det[5]) % 156
            g = 100 + (80 + 40 * det[5]) % 156
            r = 100 + (120 + 60 * det[5]) % 156
            color = (b, g, r)

            start = (int(det[0]), int(det[1])) 
            end =(int(det[2]), int(det[3])) 
            cv2.rectangle(frame,start, end, color, 2)
            text = str(int(det[5])) + f' {det[4]:#.0f} score'
            cv2.putText(frame,text,
                        (start[0]+5, start[1]-5),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        color, 1)
            # cv2.imwrite(f'test_{time.time()}.jpg', frame)
        return frame


    def tracking(self):
        if self.tracker is None:
            raise "Set the tracker first"
        if np.amin(self.read) - self.last:
            diff = np.amin(self.read) - self.last
            if diff > BUF_SZ:
                self.last = np.amin(self.read)
                return
            for i in range(1,(diff+1)%BUF_SZ):
                if not self.status[(self.last+i)%BUF_SZ]:# если инференс был на одной из картинок, которые не успели показать
                    self.temp = self.last+i
                    self.frame_counter+=1
                    frame = self.frm[self.temp%BUF_SZ]
                    self.dets = self.dets_buf[self.temp%BUF_SZ]
                    self.dets = self.dets[(self.dets[:, 4] == 0) & (self.dets[:, 5] != 0)]  # person class & score != 0
                    self.dets = np.delete(self.dets, 4, 1)  # drop class column 
                    if len(self.dets):
                        # self.dets = [x1,y1,w,h,score]
                        self.dets = np.array([[i[0], i[1], i[0] + i[2], i[1] + i[3], i[4]] for i in self.dets])
                    online_targets = self.tracker.update(self.dets, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])
                    self.frame = self.tracking_post(frame, [np.append(i.tlbr, [i.score, i.track_id]) for i in online_targets])
                    if not self.frame_counter % 30:
                        self.fps = 30/(time.time() - self.begin)
                        if self.fps > self.max_fps:
                            self.max_fps = self.fps
                        self.begin = time.time()
                    cv2.imshow('frame', self.frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                if self.temp:
                    self.last = self.temp
                    self.temp = 0
        else:
            pass


    async def get_frame(self):
        while self.frame is None:
            pass
        return self.frame
    
if __name__ == "__main__":

    khadas = Khadas()

    start = time.time()
    while True:
        khadas.show()
        # if 35 > (time.time() - start) > 30:
        #     print("Changing model")
        #     khadas.upload_models("pre.py", change = True)
        
        
    
