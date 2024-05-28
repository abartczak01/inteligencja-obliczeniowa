from os import makedirs
from os import listdir
import os
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labldir in labeldirs:
        newdir = os.path.join(dataset_home, subdir, labldir)
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
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = os.path.join(dataset_home, dst_dir, 'cats', file)  # Ścieżka docelowa dla kotów
        copyfile(src, dst)
    elif file.startswith('dog'):
        dst = os.path.join(dataset_home, dst_dir, 'dogs', file)  # Ścieżka docelowa dla psów
        copyfile(src, dst)