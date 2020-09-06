import math
import os
import random

path = './'
file_extension = '.tfrecords'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if file_extension in file:
            files.append(os.path.join(r, file))

random.shuffle(files)

i = 0
leading_zero = int(math.log10(len(files)) + 0.5)
for f in files:
    sep_index = f.rindex('/')
    new_name = f[:sep_index+1] + 'train_' + str(i).zfill(leading_zero) + '_' + f[sep_index+1:]
    print(new_name)
    os.rename(f, new_name)
    i += 1