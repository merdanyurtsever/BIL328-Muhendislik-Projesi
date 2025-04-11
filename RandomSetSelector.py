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
dataset_path = 'fma_small'
# Define the path to the output folder
output_path = 'random_dataset'
# Define the genres to select from
