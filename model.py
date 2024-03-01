import os
import tensorflow as tf
tfkl = tf.keras.layers
import numpy as np
import random
import pandas as pd
import seaborn as sns
import math
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


Xmin= pd.Series({
      'Sponginess' : -0.972080,
      'Wonder level' : -8.450380,
      'Crunchiness' : -34.802881,
      'Loudness on impact' : -9.028060,
      'Meme creativity' : -1.275860,
      'Soap slipperiness' : -6.006860,
      'Hype root' : -38.816760
    },dtype='float32')
Xmax= pd.Series({
      'Sponginess' : 15.106680,
      'Wonder level' : 18.122899,
      'Crunchiness' : 41.138801,
      'Loudness on impact' : 14.035980,
      'Meme creativity' : 6.056360,
      'Soap slipperiness' : 77.371620,
      'Hype root' : 31.024420
    },dtype='float32')

telescope=864
reg_telescope = 864
window= 200
stride= 20
#epochs=100

class model:

  def __init__(self, path):
    self.model = tf.keras.models.load_model(os.path.join(path, 'DirectForecastingFINAL'))

#simpleRNN ,window 200,stride 20,telescope 864 , batch size 64

  def predict(self, X):

    # Insert your preprocessing here
    future = X[-window:]
    #print(Xmin.shape)
    #print(future.shape)
    future = (future - Xmin) / (Xmax - Xmin)
    #print("reached here")
    future = np.expand_dims(future, axis=0)
    

    reg_predictions = np.array([])
    X_temp = future
    for reg in range(0,reg_telescope,telescope):
      pred_temp = self.model.predict(X_temp)
      if(len(reg_predictions)==0):
        reg_predictions = pred_temp
      else:
        reg_predictions = np.concatenate((reg_predictions,pred_temp),axis=1)
      X_temp = np.concatenate((X_temp[:,telescope:,:],pred_temp), axis=1)

    # Insert your postprocessing here	
    out = tf.convert_to_tensor(np.squeeze(reg_predictions,axis=0))
    out = (out*(Xmax-Xmin))+Xmin

    return out

