#!/usr/bin/env python
# coding: utf-8

# # <u><center> A terminal application that demonstrates face unlock using Keras, OpenCV. </center></u>
# ## <center><u> By Aadit Kapoor </u></center>
# 
# ### <u>Problem statement:</u> We want to create a face classifier/detector command line application that can build a person's face dataset and automatically and then can save the model.
# ### <u>Using dataset:</u> Custom made dataset of the person.
# ### <u> Face classifier datapoints: </u> haarcascade_frontalface_default.xml

# ## Importing libraries

# In[6]:


# importing
import numpy as np
import cv2
import os
import imutils
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
import keras
import matplotlib.pyplot as plt
from keras import Sequential, optimizers, losses
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D
from keras.metrics import K
import keras.backend as K
from keras.preprocessing.image import load_img, array_to_img, img_to_array, ImageDataGenerator
import random



# ## Data gathering functions
# - gather data using opencv and save it to a directory

# In[3]:


"""
    take_photos_and_save()
    Take multiple photos and save it to a folder
"""

def take_photos_and_save(folder_name="train", number_of_images=100):
    cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
    # return a single frame in variable `frame`

    if os.path.isdir(folder_name):
        print ("Folder exists!")
        print("setting random name")
        folder_name="train"+str(random.randint(0,100))
        os.mkdir(folder_name)

    else:
        print ("Creating folder: ", folder_name)
        os.mkdir(folder_name)
    
    for i in range(1, number_of_images+1):
        ret,frame = cap.read()
        frame = cv2.resize(frame, (500,500))
        #cv2.imshow('img1',frame) #display the captured image 
        cv2.imwrite(f'{folder_name}/'+str(i)+".png",frame)
        print ("saved: " + f'{folder_name}/'+str(i)+".png")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        

    cap.release()


# ## Detecting functions
# -- Detect face in the gathered data and to be later used by the prediction functions.

# In[12]:


"""
    detect_face()
    Detect a face in a given image
    return a cropped gray scale image of the face detected
"""
#loading classifier
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_face(img,is_url=False):
    cropped_img = None
    if is_url:
        img = imutils.url_to_image(img)
    print ("Shape of the image: ", img.shape)
    plt.imshow(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_img, cmap="gray")
    faces = classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    if (len(faces) == 0):
        print ("no face found!")
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cropped_img = gray_img[y:y+h, x:x+w] # the cropped image
            cropped_img = cv2.resize(cropped_img, (80,80)) # this is the input size (80,80,1)
            cropped_img = cropped_img.reshape(80,80,1)
        # perform image classification here
        
    print("Detected faces: ", len(faces))
    return cropped_img
    plt.imshow(img)


# ## Building dataset functions
# - Creates the dataset in the form of dict[images, labels] and also features some helpful functions.

# In[13]:


"""
    add_noise(numpy_matrix)
    add random noise to the image
"""
def add_noise(img):
    img = img + 3 * img.std() * np.random.random(img.shape)
    return img


"""
    create_face_dataset(string, int)
    generate the user face dataset and return it in the dict form.
"""
def create_face_dataset(img_dir, number_of_images=100): # img_dir contains all the labels with 1
    #create wrong dataset images by augumenting the images
    data = {'images':[], "labels":[]}
    # augumenting more images in img_dir
   
    for image in os.listdir(img_dir):
        if (image == ".DS_Store"):
            continue
        img = cv2.imread(img_dir+"/"+image)
        face = detect_face(img)
        if face is None:
            print ("skipped")
            continue
        data['images'].append(face)
        data['labels'].append(1)
    
    for i in os.listdir(img_dir):
        print ("noise images")
        if (i == ".DS_Store"):
            continue
        img = cv2.imread(img_dir+"/"+i)
        face = detect_face(img)
        if face is None:
            print ("skipped")
            continue
        noise_img = add_noise(face)
        data['images'].append(noise_img)
        data['labels'].append(0)
        
        
    return data


# ## Model Building functions
# - Build the model, prepare data and starting training functions.

# In[85]:


def build_model():
    model = Sequential()
    model.add(Conv2D(32, 3, padding="same",input_shape=(80,80,1), activation="relu"))
    model.add(MaxPool2D(2))
    model.add(Conv2D(64, 3, activation="relu"))
    model.add(MaxPool2D(2))
    model.add(Conv2D(128, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="rmsprop", loss=keras.losses.categorical_crossentropy, metrics=['acc'])
    return model

def prepare_data(data):
    features_train, features_test, labels_train, labels_test = train_test_split(data['images'], data['labels'], shuffle=True)
    features_train = np.array(features_train)
    features_test = np.array(features_test)
    
    # one hot encoding
    labels_train = to_categorical(labels_train)
    labels_test = to_categorical(labels_test)

    return (features_train, features_test, labels_train, labels_test)

def start_training(epochs=10, save_model=False):
    
    # build model
    model = build_model()
    
    data = create_face_dataset("train",200)
    features_train, features_test, labels_train, labels_test = prepare_data(data)
    
    # scaling data to [0-1]
    features_train = features_train * 1/255
    features_test = features_test * 1/255
    
    hist = model.fit(features_train, labels_train, epochs=epochs)
    
    # save model
    if save_model:
        model.save("face_unlock.h5")
    else:
        pass
    
    losses = hist.history['losses']
    sns.linplot(x=list(range(epochs)),y=losses, data=pd.DataFrame(losses))
    


# ## Prediction functions
# - Features functions used to predict a face in a given image using the given model.

# In[90]:


"""
    unlock function
    predicts the desired image. (required save model)
"""
def unlock(model, image):
    face_in80by80 = detect_face(image) # image shape in (80,80,1)
    prediction = model.predict(face_in80by80.reshape(1, 80, 80,1)).argmax(axis=0)[0]
    if prediction == 0:
        print ("incorrect face detected!")
    else:
        print ("correct face detected! Welcome")
    

