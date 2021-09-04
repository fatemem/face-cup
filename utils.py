import os
import os 
import random
import numpy as np
#from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from sklearn.utils import shuffle
import tensorflow.keras.backend as K
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.layers import *
from keras.models import *
from keras_layer_normalization import LayerNormalization
import random
import pandas as pd
import matplotlib.pyplot as pyplot
from itertools import repeat 
from keras_vggface.utils import preprocess_input
from numpy import asarray
import keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop,Adam,SGD
import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

batch_size = 64
LR = 0.25
alpha = 0.2


def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def triplet_loss(x, alpha = 0.2):
    # Triplet Loss function.
    anchor,positive,negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss

def embedding_model():
    vggface = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
    for layer in vggface.layers:
        layer.trainable = False
    last_layer = vggface.get_layer('avg_pool').output
    out = Dense(128, activation=None, name='out')(last_layer)
    #out=Lambda(lambda  x: K.l2_normalize(x,axis=-1))(out)
    out= LayerNormalization( name='outnorm')(out)
    encoder=Model(vggface.input,out)
    return encoder

def embedding_model1():
    densenet=tf.keras.applications.DenseNet121(include_top=False,weights="imagenet" ,input_shape=(224,224,3))
    for layer in densenet.layers:
        layer.trainable = False
    #last_layer = densenet.layers[-1].output
    #out = Dense(128, activation='relu', name='out')(last_layer)
    encoder=Model(densenet.input,densenet.output)
    return encoder

def complete_model(base_model,inp_shape):
    # Create the complete model with three
    # embedding models and minimize the loss 
    # between their output embeddings
    input_1 = Input(shape=inp_shape)
    input_2 = Input(shape=inp_shape)
    input_3 = Input(shape=inp_shape)
        
    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)
    
   
    loss = Lambda(triplet_loss)([A, P, N]) 
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss, optimizer=RMSprop())
    return model

def load_dataset(csv_name,rootdic):
    X=[]
    Y=[]
    info=pd.read_csv(csv_name,sep="\t")
    filename=rootdic+info['file name']
    for i in range(len(info)):
        x= pyplot.imread(filename[i])
        # plt.imshow(x)
        # plt.show()
        X.append(np.asarray(x,'float32'))
        # plt.imshow(X[i])
        # plt.show()
        Y.append(info.loc[i,'id']-1)
     
    #print(np.shape(X),np.shape(Y))
    X=preprocess_input(X, version=2)
    return X,Y

csv_name='F:\\mostafaie\\facecup\\dataset\\competition\\a.txt'
rootdic="F:\\mostafaie\\facecup\\dataset\\competition\\renamed sample images-v2\\aligned-2\\"
X,Y=load_dataset(csv_name,rootdic)

Y=np.asarray(Y)
#(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.25, random_state=42)
X_train=X
X_test=X
y_train=Y
y_test=Y
def get_image(label, test=False):
    """Choose an image from our training or test data with the
    given label."""
    if test:
        y = y_test; X = X_test
    else:
        y = y_train; X = X_train
    idx = np.random.randint(len(y))
    
    find_ind=[]
    for i in range(len(y)):
        if y[i]==label:
            find_ind.append(i)
    
    #if there is no data for label in train,check the test
    if len(find_ind)==0:
        return get_image(label,not test)
    #else
    idx = np.random.randint(len(find_ind))
    return X[find_ind[idx]]
 
def get_triplet(test=False):
    
   
    """Choose a triplet (anchor, positive, negative) of images
    such that anchor and positive have the same label and
    anchor and negative have different labels."""
    n = a = np.random.randint(120)
    while n == a:
        # keep searching randomly!
        n = np.random.randint(120)
        #print(n)
    a, p = get_image(a, test), get_image(a, test)
    n = get_image(n, test)
    return a, p, n

def generate_triplets(test=False):
    """Generate an un-ending stream (ie a generator) of triplets for
    training or test."""
    while True:
        list_a = []
        list_p = []
        list_n = []

        for i in range(batch_size):
            #print(i)
            a, p, n = get_triplet(test)
            # plt.imshow(a)
            # plt.text(50, 50, 'A', bbox=dict(fill=False, edgecolor='red', linewidth=20))
            # plt.show()
            # plt.imshow(p)
            # plt.text(50, 50, 'P', bbox=dict(fill=False, edgecolor='red', linewidth=20))
            # plt.show()
            # plt.imshow(n)
            # plt.text(50, 50, 'N', bbox=dict(fill=False, edgecolor='red', linewidth=20))
            # plt.show()
            list_a.append(a)
            list_p.append(p)
            list_n.append(n)
            
        A = np.array(list_a, dtype='float32')
        P = np.array(list_p, dtype='float32')
        N = np.array(list_n, dtype='float32')
        # a "dummy" label which will come in to our identity loss
        # function below as y_true. We'll ignore it.
        label = np.ones(batch_size)
        yield [A, P, N], label
        