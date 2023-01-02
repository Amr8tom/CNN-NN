from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import glob as gb
from keras.models import Sequential
import cv2
from tqdm import tqdm
from google.colab import drive
from google.colab.patches import cv2_imshow
from cv2 import *
import numpy as np
!pip install split-folders
import splitfolders
from splitfolders.split import ratio
from keras import layers,datasets,models
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import scipy
import  matplotlib.pyplot as plt
from google.colab import drive
from google.colab.patches import cv2_imshow
import tensorflow as tf
import sklearn
import skimage
from skimage.transform import resize
from sklearn.naive_bayes import GaussianNB
import glob
from sklearn import metrics
from sklearn.metrics import accuracy_score
from PIL import Image
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tensorflow import keras
from keras.utils import to_categorical
from keras.layers import Dense,Conv2D,MaxPooling2D,Activation,Flatten,Dropout
from keras.models import Sequential, load_model
from keras import layers,datasets,models
import tensorflow as tf
from skimage.io import imread

# my path of data
path="/content/drive/MyDrive/Dataset/"
# MY_name of splited data
data="splited_data"

# function to split data into 3 parts val & train & test
def split_3_parts(path_of_data,name_of_data,trainsize=0.7,valsize=0.1,testsize=0.2):
  splitfolders.ratio(input=path_of_data,output=str(name_of_data),ratio=(trainsize,valsize,testsize),seed=1333)
# calling the split
split_3_parts(path,data,trainsize=0.7,valsize=0.1,testsize=0.2)
# path of each split
path_train="/content/splited_data/train"
path_test="/content/splited_data/test"
path_val="/content/splited_data/val"
#function to images into flatten array
def load_data(folder):
    images = []
    Y=[]
    ListOfimages=[]
    for foldername in os.listdir(folder):
        ListOfimages=os.listdir(path+"/"+"{}".format(foldername))
        for image_filename in range(len(ListOfimages)):
                img_file = cv2.imread(os.path.join(folder,foldername,ListOfimages[image_filename]))
                if img_file is not None:
                     img_file=cv2.resize(img_file,(50,50))
                     images.append(img_file)
                     Y.append(int(foldername))
    images=np.array(images)
    Y=np.array(Y)
    return images,Y

# function to normalize images and convert it from rbg to gray
def normaize_and_convert_to_gray(images):
    gray_normalized_image=[]
    for i in range(len(images)):
        temp_image=cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        temp_image=(temp_image/255)
        gray_normalized_image.append(temp_image)
    gray_normalized_image=np.array(gray_normalized_image)
    return gray_normalized_image
# normalize function spicial for RBG images
def Normalize_for_rbg(images):
  rbg_normalized_images=[]
  for i in range(len(images)):
    avg=np.mean(images[i])
    temp_image=images[i]-avg
    temp_image=temp_image.astype('float32')/255.
    rbg_normalized_images.append(temp_image)
  rbg_normalized_images=np.array(rbg_normalized_images)
  return rbg_normalized_images

# loadind test & train data
X_train, Y_train = load_data(path_train)
X_test, Y_test = load_data(path_test)
X_val, Y_val = load_data(path_val)
# هذه الفانكشن للاتنين نوع من الصور يا معيدتنا يا قمر لل RBG وال GRAY 
# let Sec_NN true for build Two NN
# make type Gray To apply Cnn in Gray images data
def learn_model (X_train,Y_train,X_test,Y_test,X_val,Y_val,type="rbg",Sec_NN=False):
  type=str(type).upper()
  if(type=="RBG"):
    #normalized_data
    X_train=Normalize_for_rbg(X_train)
    X_test=Normalize_for_rbg(X_test)
    X_val=Normalize_for_rbg(X_val) 
    # learn model
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='sgd',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=200,validation_data=(X_val, Y_val))
    y_pred = model.predict(X_test).argmax(axis=1)
    print("Accuracy:  %.2f%%" % (metrics.accuracy_score(Y_test, y_pred)*100))
    print ('Recall:', recall_score(Y_test,  y_pred, average='micro'))
    print ('Precision:', precision_score(Y_test, y_pred, average='micro'))
  elif(type=="GRAY"):
    #normalized_data
    X_train=normaize_and_convert_to_gray(X_train)
    X_test=normaize_and_convert_to_gray(X_test)
    X_val=normaize_and_convert_to_gray(X_val) 
    # learn model
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(50, 50, 1)))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(46, activation='relu'))
    model.add(layers.Dense(42, activation='relu'))
    model.add(layers.Dense(40, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='sgd',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=500,validation_data=(X_val, Y_val))
    y_pred = model.predict(X_test).argmax(axis=1)
    print("Accuracy_for_2_NN  %.2f%%" % (metrics.accuracy_score(Y_test, y_pred)*100))
    print ('Recall:', recall_score(Y_test,  y_pred, average='micro'))
    print ('Precision:', precision_score(Y_test, y_pred, average='micro'))
    if(Sec_NN==True):
      print("--------------------------------the secound NN ----------------------------------")
      model2 = models.Sequential()
      model2.add(layers.Flatten(input_shape=(50, 50, 1)))
      model2.add(layers.Dense(50, activation='relu'))
      model2.add(layers.Dense(45, activation='relu'))
      model2.add(layers.Dense(40, activation='relu'))
      model2.add(layers.Dense(35, activation='relu'))
      model2.add(layers.Dropout(rate=0.5))
      model2.add(layers.Dense(10, activation='softmax'))
      model2.compile(optimizer='sgd',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
      model2.fit(X_train, Y_train, epochs=400,validation_data=(X_val, Y_val))
      y_pred = model2.predict(X_test).argmax(axis=1)
      print("Accuracy:  %.2f%%" % (metrics.accuracy_score(Y_test, y_pred)*100))
      print ('Recall:', recall_score(Y_test,  y_pred, average='micro'))
      print ('Precision:', precision_score(Y_test, y_pred, average='micro'))
      print("Fscore : ",metrics.top_k_accuracy_score)
#calling the model
#  model for gray_images   frist =  Accuracy: 80.42%       secound  = Accuracy:  90.69%
#learn_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,type="gray")
learn_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,X_val=X_val,Y_val=Y_val,type="gray",Sec_NN=True)
#  model for rgb images  Accuracy:  96.79%  remove comment to see RBG result
#learn_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,X_val=X_val,Y_val=Y_val,type="rgb")
#####################
Categories=['Cars','Ice cream cone','Cricket ball']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir='/content/drive/MyDrive/ML' 
#support_vector_machin
def support_vector_machine(X_train,X_test,Y_train,Y_test):
  X_train=normaize_and_convert_to_gray(X_train)
  X_test=normaize_and_convert_to_gray(X_test)
  X_test=X_test.flatten
  X_train=X_train.flatten
  # Instantiate the Support Vector Classifier (SVC)
  svc = SVC(C=1.0, random_state=1, kernel='linear')
  # Fit the model
  svc.fit(X_train,Y_train)
  y_predict = svc.predict(X_test)

#support_vector_machine(X_train,X_test,Y_train,Y_test)