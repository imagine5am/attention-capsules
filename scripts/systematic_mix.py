import os
import random

from numpy.random import choice
from math import ceil, log

def list_tfrecords(path, file_extension='.tfrecords'):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_extension in file:
                files.append(os.path.join(r, file))
    # files = sorted([x[::-1] for x in files])
    # files = [x[::-1] for x in files]
    return sorted(files)


def generator(list_):
    for item in list_:
        yield item
        

def output_to_file(filename, list_):
    fout = open(filename, "w")
    for item in list_:
        fout.write(item)
        fout.write("\n")
    fout.close()
    

def overlay_files(gen1, num1, gen2, num2):
    target_ratio = num1 / num2
    result = [next(gen2)]
    
    count_1 = 0
    count_2 = 1
    
    for i in range(num1 + num2 - 1):
        if count_1 < num1 and count_2 < num2:
            if count_1 / count_2 < target_ratio:
                result.append(next(gen1))
                count_1 += 1
            else:
                result.append(next(gen2))
                count_2 += 1
        elif count_1 < num1:
            result.append(next(gen1))
            count_1 += 1
        elif count_2 < num2:
            result.append(next(gen2))
            count_2 += 1
        else:
            print('There is some problem in iteration', str(i), '.')
            
    if not next(gen1, None): print('gen1 has no files.')
    if not next(gen2, None): print('gen2 has no files.')
    
    return result
    

def get_icdar15_files():
    icdar15_loc = '/mnt/data/Rohit/ACMData/4aicdarcomp/datasetoverlappingF/train_tf_records/'
    icdar15_cropped_loc = '/mnt/data/Rohit/ACMData/4aicdarcomp/datasetoverlappingF/crop_train_tf_records/'  
    icdar15_recs = list_tfrecords(icdar15_loc)
    icdar15crop_recs = list_tfrecords(icdar15_cropped_loc)
    icdar15_recs += icdar15crop_recs
    random.shuffle(icdar15_recs)
    return icdar15_recs
    
    
def get_real_files():  
    real123_loc = '/mnt/data/Rohit/ACMData/tftrainallFinal/mixed_data/mix1_ready/'
    real123_recs = list_tfrecords(real123_loc)
    
    return real123_recs


def get_synth_files_mix():
    npr_loc = '/mnt/data/Rohit/ACMData/5aSynthVideosE2E/TrainingDataRecordsvideosNPR/'
    synth_loc = '/mnt/data/Rohit/ACMData/5aSynthVideosE2E/TrainingDataRecordsvideos/'
    npr_recs = list_tfrecords(npr_loc)
    synth_recs = list_tfrecords(synth_loc)
    npr_gen = generator(npr_recs)
    synth_gen = generator(synth_recs)
    
    return overlay_files(npr_gen, len(npr_recs), synth_gen, len(synth_recs))
              
   
def rename_files(files):
    num_files = len(files)
    trailing_zeros = ceil(log(num_files, 10))
    result = []
    for idx, file in enumerate(files):
        head_tail = os.path.split(file)
        file = head_tail[1]
        file = str(str(idx).zfill(trailing_zeros)) + '_' + file
        dir = head_tail[0]
        result.append(os.path.join(dir, file))
    return result

   
if __name__ == "__main__":
    
    real_files = get_real_files()
    icdar15_files = get_icdar15_files()
    ordered_files = overlay_files(generator(real_files), len(real_files), generator(icdar15_files), len(icdar15_files))
    
    synth_files = get_synth_files_mix()
    ordered_files = overlay_files(generator(ordered_files), len(ordered_files), generator(synth_files), len(synth_files))
    
    renamed_ordered_files = rename_files(ordered_files)
    
    result = []
    for src, dest in zip(ordered_files, renamed_ordered_files):
        os.rename(src, dest)
        result.append(src + ' -> ' + dest)
        
    output_to_file('rename_result.txt', result)
