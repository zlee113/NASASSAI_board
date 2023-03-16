#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow_io as tfio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pathlib
import os
import numpy as np
from scipy import signal
from scipy.io.wavfile import read
import random
TF_CPP_MIN_LOG_LEVEL="2"


# In[10]:


def toSpectrogram(samples, sample_rate):
    frequencies, times, spectrogram = signal.spectrogram(samples[0:191786], sample_rate)
    return spectrogram

def wav2set(filename):
    file = tf.io.read_file(filename)
    signal, sample_rate = tf.audio.decode_wav(file, desired_channels = 1)
    return signal, sample_rate
    
def preprocess(filename, label):
        #READING SIGNAL FROM WAV FILES
    wav, rate = wav2set(filename)
    
        #Shift randomly, 2.5 sec to the right or left
    splitpoint = random.randint(0, 159822) - 79911
    if splitpoint < 0:
        splitpoint = 191786 + splitpoint
    firstwav = wav[:splitpoint]
    secondwav = wav[splitpoint:]
    wav = tf.concat([secondwav, firstwav], axis=0)
    
    spectrogram = tf.abs(tf.signal.stft(tf.transpose(wav[:191786]), frame_length=256, frame_step=256))
    #spectrogram = tf.expand_dims(spectrogram, axis=2)
    spectrogram = tf.squeeze(tf.transpose(spectrogram))
    
    #try adding log
    #print("spectrogram\n", spectrogram)
    return spectrogram, label

def err(y1, y2):
    Size = y1.size
    y2_i = [1 if p > 0.5 else 0 for p in y2]
    e = 0
    for i in range(Size):
        if y1[i] != y2_i[i]:
            e += 1
    return e, Size


# In[11]:


background_dir = os.path.join('/mnt','efs','fs2', 'data bucket', 'background')
whistler_dir = os.path.join('/mnt','efs','fs2', 'data bucket', 'whistler')
riser_dir = os.path.join('/mnt','efs','fs2', 'data bucket', 'riser')

#saveB = wav2set(background_dir)
backgrounds_list = tf.data.Dataset.list_files(background_dir + '/*.wav')
whistlers_list = tf.data.Dataset.list_files(whistler_dir + '/*.wav')
risers_list = tf.data.Dataset.list_files(riser_dir + '/*.wav')

num_whist = len(whistlers_list)
num_back = len(backgrounds_list)
num_rise = len(risers_list)

whistlers = tf.data.Dataset.zip((whistlers_list, tf.data.Dataset.from_tensor_slices(tf.ones(num_whist))))
nonwhistlers = tf.data.Dataset.zip((backgrounds_list, tf.data.Dataset.from_tensor_slices(tf.zeros(num_back))))
risers = tf.data.Dataset.zip((risers_list, tf.data.Dataset.from_tensor_slices(tf.zeros(num_rise))))

# fil, lab = whistlers.as_numpy_iterator().next()
# spec, lab = preprocess(fil, lab)

# plt.imshow(np.log(spec))
# plt.show()


# In[12]:


autotune = tf.data.experimental.AUTOTUNE
batch_size = 8
epoch_num = 5

data = whistlers.concatenate(nonwhistlers).concatenate(risers).shuffle(num_whist + num_back + num_rise)
data = data.map(preprocess)
data.cache() # < 250 MB
#data.shuffle(buffer_size=num_whist + num_back)
data = data.batch(batch_size)
data.prefetch(4)
# Potential Prefetch method

train_split = int((num_rise + num_whist + num_back) * 1 / 3)
test_split = num_rise + num_whist + num_back - train_split

training = data.take(train_split)
testing = data.skip(train_split).take(test_split)


# In[13]:


CP = keras.models.Sequential()
CP.add(layers.Flatten(input_shape=(129, 749)))
CP.add(layers.Dense(100, activation='relu'))
CP.add(layers.Dense(1, activation='sigmoid'))
CP.compile(
  optimizer='adam',
  loss='BinaryCrossentropy',
  metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])


# In[15]:


hist = CP.fit(training, epochs=epoch_num)#, validation_data=testing)

# npit = testing.as_numpy_iterator()
# total = 0
# wrong = 0
# for i in range(np.floor(test_split / batch_size)):
#     xin, yin = npit.next()
#     yout = CP.predict(xin)
#     error = err(yin, yout)
#     wrong += error[0]
#     total += error[1]
# print('Manual Error Rate:', wrong, "out of ", total)

# plt.plot(hist.history['loss'], 'b')
# plt.show()


# In[16]:


model_path = "/mnt/efs/fs2/zleetestbox"
# SAVED MODEL METHOD 
#tf.saved_model.save(CP, model_path)

# KERAS MODEL METHOD
converter = tf.lite.TFLiteConverter.from_keras_model(CP)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)


# In[ ]:




