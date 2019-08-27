## README
Motivated by various experiments in production and perception of non-verbal cues to deception, this is an attempt to create a lie-detector based on:
1. [Gesture](#1-gesture)
2. [Facial Expressions](#2-faces)
3. [Speech disturbances](#3-disfluency)
4. [Speech prosody](#4-prosody)

## Pre-requisite installs  
Dlib [see](https://www.learnopencv.com/install-dlib-on-ubuntu/)  
Opencv2 [see](https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html)  
FFmpeg  

## 1 Gesture
Currently works on a detected movement between frames of a video, and within a pre-defined range of skin tones. This measure excludes the face region. 

## 2 Faces
Currently: Detects faces and facial landmarks, and writes landmarks to datafile for each frame (I figured subsequent analysis would be easier - for me - in R).

## 3 disfluency
Audio is currently not supported. Plan is to look for silent pauses between speech-onset and speech-offset (anything else will be too complicated).

## 4 Prosody
Audio is currently not supported. Plan is to extract speech envelope and do some fancy stuff with it (haven't really thought about it at all!).

# To test:
>python lie_capture_beta.py --video test.mp4
