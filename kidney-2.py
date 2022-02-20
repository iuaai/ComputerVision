import numpy as np
import os 
import cv2
import dicom

#my addition
#import tensorflow as tf

import matplotlib.pyplot as plt

DATADIR = "D:/gulia/kid"
CATAGO = ["water","InPhase","CoR"]

for cat in CATAGO:
    path = os.path.join(DATADIR,cat)   # data path
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))  # import + convert to gray scale 
        img_size= 70
        img_array = cv2.resize(img_array, (img_size, img_size)) # resizeing image 
        plt.imshow(img_array, cmap = "gray")
        plt.show()
        break
    break


training_data = []
def create_training_data():
    for cat in CATAGO:
        path = os.path.join(DATADIR,cat) 
        class_num = CATAGO.index(cat)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))  # import + convert to gray scale 
                new_array = cv2.resize(img_array, (img_size, img_size)) # resizeing image 
                training_data.append([new_array, class_num]) #new_array = images[features] , class_num=labels
             
            except Exception as e:
                pass

create_training_data()
    



import random
random.shuffle(training_data)
X = []
Y = []
for features,labels in training_data:
  X.append(features)
  Y.append(labels)
X=np.array(X).reshape(-1, img_size, img_size, 3) #-1 MANY FEATURES ,1 FOR GRAYSCALE 3 FOR RGB



import pickle
pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close

pickle_out=open("Y.pickle","wb")
pickle.dump(Y,pickle_out)
pickle_out.close
np.save('Y.npy',Y)


import tensorflow as tf 
tf.__version__
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle 
import numpy as np 
X=pickle.load(open("X.pickle","rb"))
z=X
#  Y=pickle.load(open("Y.pickle","rb"))
Y=np.load('Y.npy')  #load Y
X=X/255
#layer one
model=Sequential()
model.add(Conv2D(64, (3,3) , input_shape= X.shape[1:]))
model.add(Activation("relu"))  #passing activ
model.add(MaxPooling2D(pool_size=(2,2)))

#second layer
model.add(Conv2D(64,(3,3) )) # 3 by3 window 
model.add(Activation("relu")) #passing activ
model.add(MaxPooling2D(pool_size=(2,2))) # 2 by 2 pooling window to pass to next layer

#final dense layer
model.add(Flatten()) #flatten 2d data to 1d
model.add(Dense(64))
model.add(Activation("relu"))

 #out put layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(metrics=['accuracy'],loss="binary_crossentropy",
             optimizer="adam"
             )
# training the model 
model.fit(X, Y, batch_size=22, epochs=10,  validation_split=0.2) #passing data thr model  validation=out of sample data  
import tensorflow as tf 
tf.__version__
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle 
import numpy as np 
X=pickle.load(open("X.pickle","rb"))
z=X
#  y=pickle.load(open("y.pickle","rb"))
Y=np.load('y.npy')  #load y
X=X/255
#layer one
model=Sequential()
model.add(Conv2D(64, (3,3) , input_shape= X.shape[1:]))
model.add(Activation("relu"))  #passing activ
model.add(MaxPooling2D(pool_size=(2,2)))

#second layer
model.add(Conv2D(64,(3,3) )) # 3 by3 window 
model.add(Activation("relu")) #passing activ
model.add(MaxPooling2D(pool_size=(2,2))) # 2 by 2 pooling window to pass to next layer

#final dense layer
model.add(Flatten()) #flatten 2d data to 1d
model.add(Dense(64))
model.add(Activation("relu"))

 #out put layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(metrics=['accuracy'],loss="binary_crossentropy",
             optimizer="adam"
             )
# training the model 
model.fit(X, Y, batch_size=22, epochs=10,  validation_split=0.2) #passing data thr model  validation=out of sample data
###
acc = model.history["acc"]
val_acc = model.history["val_acc"]
#dd
#loss = model.history["loss"]
#val_loss = model.history["val_loss"]
opochs = range(lon(acc))
#t
plt.plot(opochs, acc)  
plt.plot(opochs, val_acc)
plt.title("training and validation loss")
plt.figure()
#ttt
plt.plot(opochs, loss)
plt.plot(opochs, val_loss)
plt.title("training and validation loss")
plt.figure()
