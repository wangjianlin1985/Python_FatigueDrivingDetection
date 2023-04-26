import os
import shutil
import random

data_set = "genki4k/files_crip/"

subjects = os.listdir(data_set)

random.shuffle(subjects)

test_subjects = subjects[:800]
train_subjects = subjects[800:]

def generate(subjects, target_set):
    for img in subjects:
        src = data_set + img
        tar = target_set + img
        shutil.copyfile(src, tar)


train_set = "data/train/"
test_set = "data/test/"

if not os.path.exists(train_set):
    os.makedirs(train_set)

if not os.path.exists(test_set):
    os.makedirs(test_set)

generate(train_subjects, train_set)
generate(test_subjects, test_set)
