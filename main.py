import os
import cv2
import imutils
import cv2
import argparse
import sys
from glob import glob
# Importing from the opencv_face_recognition module in the folder
from opencv_face_recognition import face_recognition

'''
Here we load the face recogniser
'''
face_recogniser = face_recognition()

# Create a list of training images and labels
labels = []
images = glob('./photos/*/*', recursive=True)
for filename in images:
    labels.append(filename.split(os.path.sep)[-2].title())

# Train using this dataset
face_recogniser.train(images, labels)

# Now start video feed
print('Checking camera..')

# Edit 0 to './filename.mov' to perform facial recognition on video streaming
cam = cv2.VideoCapture('./bhavyanth.MOV')
if ( not cam.isOpened() ):
    print ("no cam")
    sys.exit()
print ("cam: present")

# Video frame parameters
cv2.namedWindow('AI Project', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('AI Project', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Loop until q is pressed
while True:
    ret, frame = cam.read()
    labelled_frame = face_recogniser.recognise(frame)
    cv2.imshow('AI Project', labelled_frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
















