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
face_detected=0


#open file to write for skin movements
fskin = open(str(args["video"]+"_skin.csv"),'w')
headers=["frame","timestamp","motion","xList","yList","wList","hList"]
writerskin=csv.DictWriter(fskin,fieldnames=headers)
writerskin.writeheader()

#open file to write for faces pos
facepos_file = open(str(args["video"]+"_facepos.csv"),'w')
fpheaders=["frame","timestamp","facex","facey","facew","faceh"]
writerfacepos=csv.DictWriter(facepos_file,fieldnames=fpheaders)
writerfacepos.writeheader()

#open file to write for faces landmarks
face_file = open(str(args["video"]+"_face.csv"),'w')
fheaders=["frame","timestamp","landmarks"]
writerface=csv.DictWriter(face_file,fieldnames=fheaders)
writerface.writeheader()


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

data = {} #Make dictionary for all values

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
		writerface.writerow({"frame":frame_id,"timestamp":time_id,"landmarks":"error"})
		face_landmarks.append("error")
		#face_frames.append(frame_id)
		#face_data.append("error")
	else:
		
		for (x, y, w, h) in faces:
			r = 0.3
			#extract face to new img
			face = clean_frame[int(y-(h*r)):int(y+h+(h*r)), int(x-(w*r)):int(x+w+(w*r))]
			#draw rectangle around face in frame output (note, these are enlarged because otherwise includes top of head, neck etc.,
			cv2.rectangle(frame, (int(x-(w*r)), int(y-(h*r))), (int(x+w+(w*r)), int(y+h+(h*r))), (255, 255, 0), 2)
			writerfacepos.writerow({"frame":frame_id,"timestamp":time_id,"facex":x,"facey":y,"facew":w,"faceh":h})
			#block out face area in blurred skin frame (so that doesn't pick up as skin movement)
			#cv2.rectangle(blur_hsv, (int(x-(w*r)), int(y-(h*r))), (int(x+w+(w*r)), int(y+h+(h*r))), (255, 0, 0), -1)
			

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
					
					face_landmarks.append(landmarks_vectorised)
					#face_frames.append(frame_id)
					#face_data.append(landmarks_vectorised)
					writerface.writerow({"frame":frame_id,"timestamp":time_id,"landmarks":landmarks_vectorised})
				cv2.imshow("face",outface)
				outface2 = cv2.resize(face, (350, 350))				
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
				#skin_min = np.array(facecol_dom)-np.array([100,100,100])
				#skin_min = [x if x>0 else 0 for x in skin_min]
				#skin_min = [x if x<255 else 255 for x in skin_min]
				#skin_min = np.array(skin_min,np.uint8)
				
				#skin_max = np.array([20,255,255],np.uint8)
				#skin_max = np.array(facecol_dom)+np.array([-100,100,100])
				#skin_max = [x if x>0 else 0 for x in skin_max]
				#skin_max = [x if x<255 else 255 for x in skin_max]
				#skin_max = np.array(skin_max,np.uint8)
				
				facecol_im = np.zeros((fheight,fwidth,3), np.uint8)
				facecol_im[:,0:fwidth//2] = facecol_dom
				cv2.imshow("skincol_det",outface2)
				cv2.imshow("skincol",facecol_im)

	
			except:
				face_landmarks.append("error")
				#face_data.append("error")
				#face_frames.append(frame_id)				
	

#skin movements (is face blocked out?)
# not currently, as people put hands in front of face, and we can always do something post processing to establish whether movements are inside face region
	#threshould using min and max values
	print(skin_min)
	print(skin_max)

	tre_green = cv2.inRange(blur_hsv, skin_min, skin_max)	

#https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/	
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	tre_green = cv2.erode(tre_green, kernel, iterations = 2)
	tre_green = cv2.dilate(tre_green, kernel, iterations = 2)
	#getting object green contour
	contours_hands, hierarchy = cv2.findContours(tre_green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 	# blur the mask to help remove noise, then apply the
	# mask to the frame
	tre_green = cv2.GaussianBlur(tre_green, (3, 3), 0)
	skinmask = cv2.bitwise_and(frame, frame, mask = tre_green)

    	
	#create blank canvas
	resizing = 500.0/width
	fheight = int(resizing*height)
	fwidth = int(resizing*width)
	skin_canv = np.zeros((fheight,fwidth,3), np.uint8)
	
#set data fields to empty on first frame (overwrites face_data loop above).
	if i is 0:
		blank_canv = np.zeros((fheight,fwidth,3), np.uint8)
		face_data=[]
		skin_data=[]
		mov_data=[]
		frame_data=[]
	#draw contours for hands
    	cv2.drawContours(frame,contours_hands,-1,(0,255,255),3)
    	cv2.drawContours(skin_canv,contours_hands,-1,(0,255,255),3)

	#draw faces on
	skin_gray = cv2.cvtColor(skin_canv, cv2.COLOR_BGR2GRAY)
	skin_gray = cv2.GaussianBlur(skin_gray, (21, 21), 0)

	xList, yList, wList, hList = [], [], [], []
	# compute the absolute differences between the current frame and previous frame (any and skin)
	if i is not 0:
	# any movement at all?
		frameDelta = cv2.absdiff(prev_gray, gray)
		thresh = cv2.threshold(frameDelta, 15, 255, cv2.THRESH_BINARY)[1]
		# dilate the thresholded image to fill in holes, then find contours
		# on thresholded image
		thresh = cv2.dilate(thresh, None, iterations=2)
		(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		# loop over the contours
		frame_movement = 0
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < args["min_area"]:
				continue
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			frame_movement = 1


	#do same for skin movement
		skinDelta = cv2.absdiff(prev_skin_gray, skin_gray)
		skinthresh = cv2.threshold(skinDelta, 15, 255, cv2.THRESH_BINARY)[1]
		# dilate the thresholded image to fill in holes, then find contours
		# on thresholded image
		skinthresh = cv2.dilate(skinthresh, None, iterations=2)
		(skincnts, _) = cv2.findContours(skinthresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		# loop over the contours
		skin_movement = 0
		for c in skincnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < args["min_area"]:
				continue
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(skin_canv, (x, y), (x + w, y + h), (255, 0, 255), 2)
			skin_movement = 1
			xList.append(x)
			yList.append(y)
			wList.append(w)
			hList.append(h)
			overlay = blank_canv.copy()
			cv2.rectangle(overlay, (x+(w/2)-5, y+(h/2)-5), (x+(w/2)+5, y+(h/2)+5), (255, 255, 255), -1)
			cv2.addWeighted(overlay, 0.2, blank_canv, 1 - 0.2 , 0, blank_canv)


	# write all data, plus skin movement csv data
		mov_data.append(frame_movement)
		skin_data.append(skin_movement)
		frame_data.append(frame_id)
		face_data.append(face_landmarks)
		writerskin.writerow({"frame":i,"timestamp":time_id,"motion":str(skin_movement),"xList":xList, "yList":yList, "wList":wList, "hList":hList})
	#put text on frames.
		cv2.putText(frame, "frame_movement: {}".format(frame_movement), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(skin_canv, "skin_movement: {}".format(skin_movement), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(skin_canv, "X: {}".format(xList), (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(skin_canv, "Y: {}".format(yList), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# show the frame and various manipulations
		#cv2.imshow("Frame Delta", frameDelta)		
		cv2.imshow("Thresh", thresh)
		cv2.imshow("Frame", frame)
		#cv2.imshow("hand", blur_hsv)
		#cv2.imshow("skin_detection", skin_canv)
		cv2.imshow("skin",skinmask)
		cv2.imshow("skin movement heatmap",blank_canv)
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


#write data
data["frame"]=frame_id
data["mov_data"]=mov_data
data["skin_data"]=skin_data
data["face_data"]=face_data

