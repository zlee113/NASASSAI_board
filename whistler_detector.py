# import tensorflow_io as tfio
import tensorflow as tf
# from tensorflow import keras
# #from tensorflow.keras import layers
# #import matplotlib.pyplot as plt
# #import pathlib
# import os
import numpy as np
# from scipy import signal
# from scipy.io.wavfile import read
# #import random

# from PIL import Image

# TF_CPP_MIN_LOG_LEVEL="2"

# def wav2set(filename):
#     file = tf.io.read_file(filename)
#     signal, sample_rate = tf.audio.decode_wav(file, desired_channels = 1)
#     return signal, sample_rate
    
# def preprocess(filename, label):
#         #READING SIGNAL FROM WAV FILES
#     wav, rate = wav2set(filename)
    
#         #Shift randomly, 2.5 sec to the right or left
#     # splitpoint = random.randint(0, 159822) - 79911
#     # if splitpoint < 0:
#     #     splitpoint = 191786 + splitpoint
#     # firstwav = wav[:splitpoint]
#     # secondwav = wav[splitpoint:]
#     # wav = tf.concat([secondwav, firstwav], axis=0)
    
#     spectrogram = tf.abs(tf.signal.stft(tf.transpose(wav[:191786]), frame_length=256, frame_step=256))
#     #spectrogram = tf.expand_dims(spectrogram, axis=2)
#     spectrogram = tf.squeeze(tf.transpose(spectrogram))
    
#     #try adding log
#     #print("spectrogram\n", spectrogram)
#     return spectrogram, label


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
#quit()