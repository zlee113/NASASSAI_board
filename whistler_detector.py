# import tensorflow_io as tfio
#import tensorflow as tf
# from tensorflow import keras
# #from tensorflow.keras import layers
# #import matplotlib.pyplot as plt
# #import pathlib
# import os
import numpy as np
# from scipy import signal
# from scipy.io.wavfile import read
# #import random
import tflite_runtime.interpreter as tflite
from scipy import signal
from scipy.io.wavfile import read

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


def model_run(filename):
    # create spectrogram from wav file
    sample_rate, samples = read(filename)
    frequencies, times, spectrogram = signal.spectrogram(samples[:191786], sample_rate, noverlap=0)
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="model_v2_edgetpu.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], [spectrogram])

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Whistler Inference:", output_data[0][0])

print("Run with whistler:")
model_run("whistler0.wav")
print("Run with background noise:")
model_run("whistler.wav")





#quit()

# class TestModel(tf.Module):
#   def __init__(self):
#     super(TestModel, self).__init__()

#   @tf.function(input_signature=[tf.TensorSpec(shape=[1, 10], dtype=tf.float32)])
#   def add(self, x):
#     '''
#     Simple method that accepts single input 'x' and returns 'x' + 4.
#     '''
#     # Name the output 'result' for convenience.
#     return {'result' : x + 4}


# SAVED_MODEL_PATH = 'content/saved_models/test_variable'
# TFLITE_FILE_PATH = 'model.tflite'

# # Save the model
# module = TestModel()
# # You can omit the signatures argument and a default signature name will be
# # created with name 'serving_default'.
# tf.saved_model.save(
#     module, SAVED_MODEL_PATH,
#     signatures={'my_signature':module.add.get_concrete_function()})

# # Convert the model using TFLiteConverter
# converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
# tflite_model = converter.convert()
# with open(TFLITE_FILE_PATH, 'wb') as f:
#   f.write(tflite_model)

# # Load the TFLite model in TFLite Interpreter
# interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
# # There is only 1 signature defined in the model,
# # so it will return it by default.
# # If there are multiple signatures then we can pass the name.
# my_signature = interpreter.get_signature_runner()

# # my_signature is callable with input as arguments.
# output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
# # 'output' is dictionary with all outputs from the inference.
# # In this case we have single output 'result'.
# print(output['result'])