# the purpose of this code is to select 100 random music from each of the 8 different genres and put them in a folder with the genre names

import os
import random
import shutil
import pandas as pd
import numpy as np
import glob
import librosa
import soundfile as sf

# Define the path to the dataset
dataset_path = 'D:/Datasets/GTZAN/genres_original'