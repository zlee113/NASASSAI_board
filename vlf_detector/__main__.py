import argparse
from .detector import Detector
TITLE = "Whistler Detector"
MODEL_PATH = "model_v2_edgetpu.tflite"
STATION = "Todmorden"
SPEC = False
DURATION = 20
SPLIT = 4
def parse_cli():
    """ Parses and validates command line arguments """
    parser = argparse.ArgumentParser()
    parser.prog = TITLE
    parser.add_argument("-m", "--model", type=str, default=MODEL_PATH, help="Model Name to use for interpreter")
    parser.add_argument("-s", "--station", type=str, default=STATION, help="Pick station to stream live data from")
    parser.add_argument("-o", "--output", type=bool, default=SPEC, help="Produce a spectrogram of the output signals")
    parser.add_argument("-d", "--duration", type=int, default=DURATION, help="Duration of wav files you want to look at")
    parser.add_argument("-S", "--split", type=int, default=SPLIT, help="Amount of splits for the duration")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_cli()
    detector = Detector(args.model, args.station,args.output, args.duration, args.split)
    detector.run()