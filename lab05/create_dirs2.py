from os import makedirs
from os import listdir
import os
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = 'input/'
subdirs = ['train/', 'test1/']
for subdir in subdirs:
    newdir = os.path.join(dataset_home, subdir)
    makedirs(newdir, exist_ok=True)

# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25

# copy training dataset images into subdirectories
src_directory = '../dogs-cats-mini/'
for file in listdir(src_directory):
    src = os.path.join(src_directory, file)  # Pełna ścieżka do pliku
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test1/'
    dst = os.path.join(dataset_home, dst_dir, file)
    copyfile(src, dst)