import numpy as np
import tflite_runtime.interpreter as tflite
from scipy import signal
from scipy.io.wavfile import read
import multiprocessing as mp
import os
from datetime import datetime
import time

class Detector:

    def __init__(self, model: str, station: str, duration: float, split: int):

        # Creates an interpreter based on the model given
        self.interpreter = tflite.Interpreter(model_path="./" + model)

        # Dictionary of all the stations found on vlfrx
        stations = {"Todmorden": "5.9.106.210,4401", 
                    "Cumiana": "5.9.106.210,4415", 
                    "Surfside Beach": "5.9.106.210,4434",
                    "Forest": "5.9.106.210,4435",
                    "Warsaw": "5.9.106.210,4438",
                    "Heathcote": "5.9.106.210,4439",
                    "Heidelberg": "5.9.106.210,4441"
                    }
        self.station = stations[station]
        self.duration = duration
        self.split = split

        # Makes appropriate directories based off of usage in the code
        os.system("mkdir -p ./detector/tmp")
        os.system("mkdir -p ./buffer")

    # Run function holds all the calls in an infinite loop to always be discovering whistlers
    def run(self):
        try:
            while(True):
                self.generate_wav_files()
                start = time.time()
                self.process_output()
                end = time.time()
                print("Interpreter duration: " + str(end-start))
                # p = mp.Process(target=self.process_output)
                # p.start()
                # p.join()
        except KeyboardInterrupt:
            SystemExit(1)

    def generate_wav_files(self):
        # Vtvorbis command connects to station and records for a set duration
        # Vtraw is used to save that recording as a wav file
        cmd = "vtvorbis -E " + str(self.duration) + " -dn" + self.station + " | vtraw -ow > ./detector/tmp/vlfex.wav"
        os.system(cmd)
        # FFMPEG is used to split up that recording into separate files for model processing
        seg1 = 'ffmpeg -y -ss 0 -t ' + str(self.split) + ' -i ./detector/tmp/vlfex.wav ./detector/tmp/out1.wav'
        seg2 = 'ffmpeg -y -ss 4.5 -t ' + str(self.split) + ' -i ./detector/tmp/vlfex.wav ./detector/tmp/out2.wav'
        seg3 = 'ffmpeg -y -ss 9 -t ' + str(self.split) + ' -i ./detector/tmp/vlfex.wav ./detector/tmp/out3.wav'
        seg4 = 'ffmpeg -y -ss 13.97 -t ' + str(self.split) + ' -i ./detector/tmp/vlfex.wav ./detector/tmp/out4.wav'

        os.system(seg1)
        os.system(seg2)
        os.system(seg3)
        os.system(seg4)

    def process_output(self):

        # Runs each iteration through model and ends if one contains a whistler
        for i in range(4):
            val = self.model_run(f"./detector/tmp/out{i+1}.wav")
            if val > 0.75:
                now = datetime.now()
                dt = now.strftime("%d%m%Y%H%M%S")
                mv_cmd = "mv ./detector/tmp/vlfex.wav ./buffer/" + dt
                os.system(mv_cmd)
                break

    def model_run(self, filename):
        # create spectrogram from wav file
        sample_rate, samples = read(filename)
        frequencies, times, spectrogram = signal.spectrogram(samples[:191786], sample_rate, noverlap=0)

        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Test the model on input data and make sure its the right size
        input_shape = input_details[0]['shape']
        # spectrogram = np.array(spectrogram, dtype=np.float32)
        # spec_v, spec_h = np.shape(spectrogram)
        # size = np.zeros([input_shape[1], input_shape[2] - spec_h])
        # print(np.shape(size))
        # spec = np.hstack((spectrogram, size))
        # #input_data = np.array(np.zeros(input_shape), dtype=np.float32)
        # #np.copyto(input_data, spectrogram)
        # print(np.shape(spectrogram))
        self.interpreter.set_tensor(input_details[0]['index'], [spectrogram])

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        print("Whistler Inference:", output_data[0][0])
        return output_data[0][0]