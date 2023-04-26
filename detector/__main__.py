import argparse
from .detector import Detector
TITLE = "Whistler Detector"
MODEL_PATH = "model_v2_edgetpu.tflite"
STATION = "Todmorden"
SPEC = False
DURATION = 20
SPLIT = 6
def parse_cli():
    """ Parses and validates command line arguments """
    parser = argparse.ArgumentParser()
    parser.prog = TITLE
    parser.add_argument("-m", "--model", type=str, default=MODEL_PATH, help="Model Name to use for interpreter")
    parser.add_argument("-s", "--station", type=str, default=STATION, help="Pick station to stream live data from")
    parser.add_argument("-d", "--duration", type=float, default=DURATION, help="Duration of wav files you want in buffer folder, default is 20 seconds")
    parser.add_argument("-S", "--split", type=int, default=SPLIT, help="time of splits for model, default model is 6 second wav files")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_cli()
    detector = Detector(args.model, args.station, args.duration, args.split)
    detector.run()