from mtcnn.mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf
import keras.losses
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras_layer_normalization import LayerNormalization
from numpy import asarray
import pandas as pd
from PIL import Image
import io
import os
from itertools import combinations
def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

FACE_DETECTOR = MTCNN(min_face_size = 20 ,steps_threshold = [0.3, 0.4, 0.5],scale_factor=0.95)
base_model=load_model('base_model.h5',custom_objects={'identity_loss':identity_loss,'LayerNormalization':LayerNormalization},compile=False)



def preprocess(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


def face_detection(img):
    required_size=(224, 224)
    pixels = np.array(img)
    
    results = FACE_DETECTOR.detect_faces(pixels)

    if len(results) > 0:
        x1, y1, width, height = results[0]['box']
        i=0
        while x1<0 or y1<0:
            i=i+1
            if i==len(results):
                break
            x1, y1, width, height = results[i]['box']
        if x1<0 or y1<0:
            x1=0
            y1=0
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        
    else:
        face=pixels
        #print("Face could not be detected.")
        
    # resize pixels to the model size
   
    try:
        image = Image.fromarray(face)
    except:
        
        print(results)
        plt.imshow(face)
        plt.show()
   
    
    image = image.resize(required_size)
    face_array = asarray(image,'float64')

    return face_array

def match(probe, gallery):
    prob_image = face_detection(probe)
    prob_image = preprocess(prob_image)
    prob_image=np.expand_dims(prob_image, axis=0)
    prob_image_representation = base_model.predict(prob_image)

    gallery_image = face_detection(gallery)
    gallery_image = preprocess(gallery_image)
    gallery_image=np.expand_dims(gallery_image, axis=0)

    gallery_image_representation = base_model.predict(gallery_image)
    
    distance = np.linalg.norm(prob_image_representation-gallery_image_representation)
    d=float(distance) 
    s=1/(1+d)
    return s       



def read_images(dataset_ad):
    gallery_images=[]
    probe_images=[]
    names = [dataset_ad + '/' + x for x in os.listdir(dataset_ad)]
    all_data=[list([x + '/' + y for y in os.listdir(x)]) for x in names]
    probe_images_name=[all_data[i][j] for i in range(len(all_data)) for j in range(1,len(all_data[i])) ]
    gallery_images_name=[all_data[i][0] for i in range(len(all_data))]
    gallery_images=[plt.imread(name) for name in gallery_images_name ]
    probe_images=[plt.imread(name) for name in probe_images_name ]
    return gallery_images,probe_images



dataset_ad="F:/mostafaie/facecup/dataset/my dataset"
gallery_images,probe_images=read_images(dataset_ad)






for probe_image in probe_images:
    g_ind=0
    s=[]
    for gallery_image in gallery_images:
        score=match(probe_image,gallery_image)
        temp=[g_ind,score]
        g_ind=g_ind+1
        s.append(temp)
    s=sorted(s, key = lambda x: x[1]) 
    
    plt.imshow(probe_image)
    plt.show()
    
    plt.imshow(gallery_images[s[-1][0]])
    plt.show()
       