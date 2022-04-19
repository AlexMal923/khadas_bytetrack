import cv2
import time
cap = cv2.VideoCapture(0)
print("Parameters BEFORE assignment: ")
print("WIDTH: " + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("HEIGHT: " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("FPS: " + str(cap.get(cv2.CAP_PROP_FPS)))
print("FOURCC: " + str(cap.get(cv2.CAP_PROP_FOURCC)))
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

last = time.time()
try:
    while True:
        ret, frame = cap.read()
        
        print("FPS: ", int(1/(time.time() - last)))
        # print((time.time() - last)*1000, end='\r')
        last = time.time()
        # cv2.imshow('frame', frame)
        # key = cv2.waitKey(1)
        # if (key == 27):
        #     break
except KeyboardInterrupt:
    cap.release()
    cv2.imwrite("camera.jpg", frame)
