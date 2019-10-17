# import the necessary packages
import argparse
import numpy as np
from scipy.io import wavfile
import pandas as pd 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--audio", help="path to the audio file")
args = vars(ap.parse_args())

input_data = wavfile.read(args["audio"])
fs=input_data[0]
audionp = input_data[1]
#audionp = np.fromstring(audionp,"Int16")

pd.DataFrame(audionp).to_csv(str(args["audio"]+"_audio.csv"))
