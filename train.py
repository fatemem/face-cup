# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:21:44 2021

@author: Behsa PC1
"""

from utils import *

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf
import keras.losses
    




 

train_generator = generate_triplets()
test_generator = generate_triplets(test=True)
#batch= next(train_generator)

base_model = embedding_model()
model = complete_model(base_model,(224, 224, 3))
#model=load_model('best_tripletloss.h5',custom_objects={'identity_loss':identity_loss})
model.summary()
mcp_save = ModelCheckpoint('best_tripletloss12_LayerNormalization.h5', save_best_only=True, monitor='val_loss', mode='min')


history = model.fit_generator(train_generator, 
                    validation_data=test_generator, 
                    epochs=20,
                    steps_per_epoch=20,validation_steps=30,callbacks=[mcp_save])

# history = model.fit_generator(train_generator,  
#                     epochs=100,
#                     steps_per_epoch=20,callbacks=[mcp_save])

model.save('model_LayerNormalization.h5')



# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Losses',size = 20)
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.show()