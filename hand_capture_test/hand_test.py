import cv2
import numpy as np
import math
import argparse
import csv

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=20, help="minimum area size")
args = vars(ap.parse_args())


# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])


#get video length, width, height, fps etc.
length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = camera.get(cv2.CAP_PROP_FPS)


hand_file = open(str(args["video"]+"_hand.csv"),'w')
fheaders=["frame","timestamp","coords"]
writerhand=csv.DictWriter(hand_file,fieldnames=fheaders)
writerhand.writeheader()


for i in range(length):
	# grab the current frame 
	(grabbed, frame) = camera.read()
	frame_id = i
	time_id = i/fps
	print frame_id
	print i
 	# if the frame could not be grabbed, then we have reached the end of the video
	if not grabbed:
		break
        
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame
         
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # define range of skin color in HSV
        lower_skin = np.array([0, 48, 80],np.uint8)
        upper_skin = np.array([20, 255, 255],np.uint8)
     #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        
        
    #find contours
        #_,contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                
        cv2.imshow("roi",mask)
        

    
	k = cv2.waitKey(5) & 0xFF
    	if k == 27:
		break
    
cv2.destroyAllWindows()
camera.release()    
    




