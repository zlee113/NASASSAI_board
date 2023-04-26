import numpy as np
import tflite_runtime.interpreter as tflite
from scipy import signal
from scipy.io.wavfile import read
import multiprocessing as mp
import os
from datetime import datetime
import time

class Detector:

    def __init__(self, model: str, station: str, output: bool, duration: float, split: int):

        self.interpreter = tflite.Interpreter(model_path="./" + model)
        stations = {"Todmorden": "5.9.106.210,4401", 
                    "Cumiana": "5.9.106.210,4415", 
                    "Surfside Beach": "5.9.106.210,4434",
                    "Forest": "5.9.106.210,4435",
                    "Warsaw": "5.9.106.210,4438",
                    "Heathcote": "5.9.106.210,4439",
                    "Heidelberg": "5.9.106.210,4441"
                    }
        self.station = stations[station]
        self.output = output
        self.duration = duration
        self.split = split
        os.system("mkdir -p ./detector/tmp")
        os.system("mkdir -p ./buffer")


    def run(self):
        try:
            while(True):
                self.generate_wav_files("buffer")
                start = time.time()
                self.process_output()
                end = time.time()
                print("Interpreter duration: " + str(end-start))
                # p = mp.Process(target=self.process_output)
                # p.start()
                # p.join()
        except KeyboardInterrupt:
            SystemExit(1)

    def generate_wav_files(self,dir):
        cmd = "vtvorbis -E " + str(self.duration) + " -dn" + self.station + " | vtraw -ow > ./detector/tmp/vlfex.wav"
        os.system(cmd)
        seg1 = 'ffmpeg -y -ss 0 -t 6 -i ./detector/tmp/vlfex.wav ./detector/tmp/out1.wav'
        seg2 = 'ffmpeg -y -ss 4.5 -t 6 -i ./detector/tmp/vlfex.wav ./detector/tmp/out2.wav'
        seg3 = 'ffmpeg -y -ss 9 -t 6 -i ./detector/tmp/vlfex.wav ./detector/tmp/out3.wav'
        seg4 = 'ffmpeg -y -ss 13.97 -t 6 -i ./detector/tmp/vlfex.wav ./detector/tmp/out4.wav'

        os.system(seg1)
        os.system(seg2)
        os.system(seg3)
        os.system(seg4)

    def process_output(self):
        # values = []
        # values.append()
        # values.append(self.model_run("./vlf_detector/tmp/out2.wav"))
        # values.append(self.model_run("./vlf_detector/tmp/out3.wav"))
        # values.append(self.model_run("./vlf_detector/tmp/out4.wav"))
        for i in range(4):
            val = self.model_run(f"./detector/tmp/out{i+1}.wav")
            if val > 0.75:
                mv_cmd = "mv ./detector/tmp/vlfex.wav ./buffer/whistler" + str(datetime.today())
                os.system(mv_cmd)
                break
        # # Remove Outputs
        # os.system("rm ./tmp/out/out1.wav")
        # os.system("rm ./tmp/out/out2.wav")
        # os.system("rm ./tmp/out/out3.wav")
        # os.system("rm ./tmp/out/out4.wav")
        # os.system("rm ./tmp/vlfex.wav")

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