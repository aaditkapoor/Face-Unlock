# A sample command line application using the functions developed above.
# author:aadit kapoor
# This is a rough untested version of the command line app.
# The whole purpose of this script is to help users understand the intersection of machine learning and software development
# version: 1.0

import time
import random
from main import *
import os
import cv2


model = build_model()
if os.path.exists("face_unlock.h5"):
    print ("Model weights loaded")
    model.load_weights("face_unlock.h5")
else:
    print ("No model files are present!")

print


print ("Welcome to Face Unlock App (By Aadit Kapoor)")
print ("================================")
print

while True:
    print ("Select an option to begin: ")

    print ("1 - Train a face")
    print ("2 - Test a face")
    print ("3 - Exit")

    choice = str(input("Enter choice: "))

    if choice == "1":
        print ("Sit back and look at the camera.")
        print ("Using OpenCV VideoCapture(0)....")
        time.sleep(2)
        take_photos_and_save("train", number_of_images=200)
        start_training(epochs=50, save_model=True)

    elif choice == "2":
        cap = cv2.VideoCapture(0)
        for i in range(1, 2): # run only once!
            ret,frame = cap.read()
            frame = cv2.resize(frame, (800,800))
            cv2.imshow('img1',frame) #display the captured image 
            

        cap.release()

        unlock(model, frame)
    else:
        print ("Thanks for using!")
        break

