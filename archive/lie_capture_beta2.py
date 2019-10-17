# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import csv
import numpy as np
import copy
import dlib
import math
import glob
import random
import itertools

#face detection preambles
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cascPath = "haarcascade_frontalface_default.xml"

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


#set initial skin thresholds.. might need changing.
#it would be good to be able to set skin thresholds based on a click of the mouse or something. or maybe via avg pixel in face regions?
skin_min = np.array([0, 48, 80],np.uint8)
skin_max = np.array([20, 255, 255],np.uint8)
#https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/


#open file to write for skin movements
fskin = open(str(args["video"]+"_skin.csv"),'w')
headers=["frame","timestamp","motion","xList","yList","wList","hList"]
writerskin=csv.DictWriter(fskin,fieldnames=headers)
writerskin.writeheader()

#open file to write for faces pos
facepos_file = open(str(args["video"]+"_facepos.csv"),'w')
fpheaders=["frame","timestamp","face"]
writerfacepos=csv.DictWriter(facepos_file,fieldnames=fpheaders)
writerfacepos.writeheader()

#open file to write for faces landmarks
face_file = open(str(args["video"]+"_face.csv"),'w')
fheaders=["frame","timestamp","landmarks"]
writerface=csv.DictWriter(face_file,fieldnames=fheaders)
writerface.writeheader()




########################### functions
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def get_facecol(face_im,fheight,fwidth):
	outface2 = cv2.resize(face_im, (350, 350))	
	roi_size = 150 # (10x10)
	outface2 = outface2[(350-roi_size)/2:(350+roi_size)/2,(350-roi_size)/2:(350+roi_size)/2]
	#cv2.imshow("skincol",outface2)
	facecol = np.reshape(outface2, (-1,3))
	print(facecol.shape)
	#Change these values to fit the size of your region of interest
	facecol = np.float32(facecol)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS
	compactness,labels,centers = cv2.kmeans(facecol,1,None,criteria,10,flags)
	facecol_dom=centers[0].astype(np.int32)
	skin_min = np.array(facecol_dom)-np.array([100,70,30])
	skin_min = [x if x>0 else 0 for x in skin_min]
	skin_min = [x if x<255 else 255 for x in skin_min]
	skin_min = np.array(skin_min,np.uint8)
	#skin_min = np.array([0, 48, 80],np.uint8)
	skin_max = np.array([20,255,255],np.uint8)
	#skin_max = np.array([x+50 for x in facecol_dom],np.uint8)
	facecol_im = np.zeros((fheight,fwidth,3), np.uint8)
	facecol_im[:,0:width//2] = facecol_dom
	cv2.imshow("skincol_det",outface2)
	cv2.imshow("skincol",facecol_im)
	print skin_min
	print skin_max
	return skin_min, skin_max



def face_detect(frame, clean_frame, frame_id, time_id):
	#detect faces
	face_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faceCascade = cv2.CascadeClassifier(cascPath)
	faces = faceCascade.detectMultiScale(
	    face_gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30),
	    flags = cv2.CASCADE_SCALE_IMAGE
	)

	#extract faces from canvas
	#http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
	face_landmarks=[]
	if len(faces)==0:
		face_data=[frame_id, time_id, 0,0,0,0]
		landmark_data = [frame_id, time_id, "error"]
	else:
		for (x, y, w, h) in faces:
			r = 0.3
			#extract face to new img
			face = clean_frame[int(y-(h*r)):int(y+h+(h*r)), int(x-(w*r)):int(x+w+(w*r))]
			#draw rectangle around face in frame output (note, these are enlarged because otherwise includes top of head, neck etc.,
			cv2.rectangle(frame, (int(x-(w*r)), int(y-(h*r))), (int(x+w+(w*r)), int(y+h+(h*r))), (255, 255, 0), 2)
			face_data = [frame_id, time_id,[x,y,w,h]]
					

			#if there's a face, show it.
			try:

				outface = cv2.resize(face, (350, 350)) #Resize face so all images have same size
				face_gray = cv2.cvtColor(outface, cv2.COLOR_BGR2GRAY)
		    		clahe_image = clahe.apply(face_gray)
				detections = detector(clahe_image, 1) #Detect the faces in the image

				for k,d in enumerate(detections): #For each detected face
					#Get coordinates
					shape = predictor(clahe_image, d)
					lmxlist = []
					lmylist = []

					#Store X and Y coordinates in two lists
					for j in range(1,68): 
						#For each point, draw a red circle with thickness2 on the original frame
						cv2.circle(outface, (shape.part(j).x, shape.part(j).y), 1, (0,0,255), thickness=2)
						lmxlist.append(float(shape.part(j).x))
						lmylist.append(float(shape.part(j).y))

					#Find both coordinates of centre of gravity
					xmean = np.mean(lmxlist)
					ymean = np.mean(lmylist)
					#Calculate distance centre <-> other points in both axes
					xcentral = [(x-xmean) for x in lmxlist] 
					ycentral = [(y-ymean) for y in lmylist]
					landmarks_vectorised = []
				
					for x, y, w, z in zip(xcentral, ycentral, lmxlist, lmylist):
						landmarks_vectorised.append(w)
						landmarks_vectorised.append(z)
						meannp = np.asarray((ymean,xmean))
						coornp = np.asarray((z,w))
						dist = np.linalg.norm(coornp-meannp)
						landmarks_vectorised.append(dist)
						landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))

					landmark_data = [frame_id, time_id, landmarks_vectorised]
					writerface.writerow({"frame":landmark_data[0],"timestamp":landmark_data[1],"landmarks":landmark_data[2]})
				cv2.imshow("face",outface)
				skin_min, skin_max = get_facecol(face)
				print skin_min
			except:
				landmark_data = [frame_id, time_id, "error"]

	return face_data








############################# main



# loop over the frames of the video
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
 
	
	# resize the current frame,
	frame = imutils.resize(frame, width=500)	
	#make and show clean copy	
	clean_frame = copy.copy(frame)
	cv2.imshow("clean",clean_frame)
	
	#convert it to grayscale, and blur it
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	#frame = increase_brightness(frame, value=50)
	
	#blur copies for skin movement.
	skin_blur = cv2.GaussianBlur(frame,(5,5),0)
    	blur_hsv = cv2.cvtColor(skin_blur, cv2.COLOR_BGR2HSV)

	#create blank canvas
	resizing = 500.0/width
	fheight = int(resizing*height)
	fwidth = int(resizing*width)
	skin_canv = np.zeros((fheight,fwidth,3), np.uint8)
	
	face_data = face_detect(frame, clean_frame, frame_id, time_id)
	writerfacepos.writerow({"frame":face_data[0],"timestamp":face_data[1],"face":face_data[2]})	

	print skin_min

	if i is 0:
		blank_canv = np.zeros((fheight,fwidth,3), np.uint8)
		face_data=[]
		skin_data=[]
		mov_data=[]
		frame_data=[]
	#draw contours for hands
    	#cv2.drawContours(frame,contours_hands,-1,(0,255,255),3)
    	#cv2.drawContours(skin_canv,contours_hands,-1,(0,255,255),3)

	#draw faces on
	skin_gray = cv2.cvtColor(skin_canv, cv2.COLOR_BGR2GRAY)
	skin_gray = cv2.GaussianBlur(skin_gray, (21, 21), 0)

	xList, yList, wList, hList = [], [], [], []
	# compute the absolute differences between the current frame and previous frame (any and skin)
	if i is not 0:

		cv2.imshow("Frame", frame)

		# break statements
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break

#set frame for next loop
	prev_gray = gray
	prev_skin_gray = skin_gray


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()





