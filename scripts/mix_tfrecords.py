import cv2
import numpy as np
import os
import random 
import tensorflow as tf

from tqdm import tqdm


def read_examples(tfrecord_file):
    '''Returns a list of examples read from a single tfrecord'''
    examples = []
    for example in tf.python_io.tf_record_iterator(tfrecord_file):
        examples.append(tf.train.Example.FromString(example))
    return examples


def write_tfrecord(out_file, examples):
    '''Writes examples to out_files'''
    writer = tf.python_io.TFRecordWriter(out_file)
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()


def list_files(src_loc, file_extension):
    '''List files in src_loc and returns files which have the valid file_extension'''
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(src_loc):
        for file in f:
            if file.endswith(file_extension):
                files.append(os.path.join(r, file))
    return files


def get_img_and_label(example):
    '''returns the image and label feature in an Example'''
    label_str = example.features.feature['image/text'].bytes_list.value[0]
    label = label_str.decode('utf-8')
    
    img_str = example.features.feature['image/encoded'].bytes_list.value[0]
    img_np_arr = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(img_np_arr, 1)
    
    return img, label


def save_rand_image(examples, prefix):
    '''Saves a random image from an Examples list'''
    example = random.choice(examples)
    img, label = get_img_and_label(example)
    cv2.imwrite(prefix+'_'+label+'.png', img)
    

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def correct_examples(examples):
    corrected_examples = []
    for example in examples:
        img_str = example.features.feature['image/encoded'].bytes_list.value[0]
        img_np_arr = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(img_np_arr, 1)
        
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, img = cv2.imencode('.png', img_bgr)
        
        corrected_example = tf.train.Example(features=tf.train.Features(
                        feature={'image/format': example.features.feature['image/format'], 
                                'image/encoded': _bytes_feature(img.tostring()),
                                'image/class': example.features.feature['image/class'],
                                'image/unpadded_class': example.features.feature['image/unpadded_class'],
                                'image/height': example.features.feature['image/height'],
                                'image/width': example.features.feature['image/width'],
                                'image/flag_synth': example.features.feature['image/flag_synth'],
                                'image/orig_width': example.features.feature['image/orig_width'],
                                'image/text': example.features.feature['image/text'],
                                'image/charBB': example.features.feature['image/charBB'],
                                'image/wordBB': example.features.feature['image/wordBB'],
                                'image/lineBB': example.features.feature['image/lineBB'],
                                'image/transformation': example.features.feature['image/transformation'],
                                'image/video_no': example.features.feature['image/video_no'], 
                                'image/frame_no': example.features.feature['image/frame_no'],}))
        
        corrected_examples.append(corrected_example)
    return corrected_examples


if __name__=='__main__':
    tf_src_loc = '/mnt/data/Rohit/ACMData/tftrainallFinal/1trainFinal'
    # tf_src_loc = './old/'
    tf_dest_loc = './mix1/'
    image_loc = './pictures/'
    num_examples_out = 100
    leading_zero = 8
    queue_capacity = 10000
    min_after_dequeue = 7000
    
    tfrecords = list_files(src_loc=tf_src_loc, file_extension='.tfrecords')
    random.shuffle(tfrecords)
    
    queue = []
    num_tfrecords = 0
    num_examples = 0
    
    for tfrecord in tqdm(tfrecords):
        if len(queue) >= queue_capacity:
            print('Creating tfrecords.')
            while len(queue) > min_after_dequeue:
                rand_indices = random.sample(range(0, len(queue)), num_examples_out)
                rand_indices.sort(reverse=True)
                rand_examples = [queue.pop(idx) for idx in rand_indices]
                random.shuffle(rand_examples)
                
                new_name = 'mix_' + str(num_tfrecords).zfill(leading_zero)+'.tfrecords'
                write_tfrecord(os.path.join(tf_dest_loc, new_name), rand_examples)
                num_tfrecords += 1
        
        examples = read_examples(tfrecord)
        if examples:
            if 'trainsynthvidNPR' in tfrecord:
                print('Caught', tfrecord)
                examples = correct_examples(examples)
            
            num_examples += len(examples)
            queue += examples
            _, filename = os.path.split(tfrecord)
            save_rand_image(examples, os.path.join(image_loc, filename[17:-10]))
        else:
            print(tfrecord, 'is empty.')
            
        os.remove(tfrecord)
    
    print('Creating tfrecords.')
    while len(queue):
        examples = queue[:num_examples_out]
        random.shuffle(examples)
        del queue[:num_examples_out]
        
        new_name = 'mix_' + str(num_tfrecords).zfill(leading_zero)+'.tfrecords'
        write_tfrecord(os.path.join(tf_dest_loc, new_name), examples)
        num_tfrecords += 1
        
    print('Number of Example:', num_examples)
    print('Number of tfrecords:', num_tfrecords)
            
    