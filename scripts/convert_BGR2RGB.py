import cv2
import numpy as np
import os
import tensorflow as tf

def read_examples(tfrecord_file):
    '''Returns a list of examples read from a single tfrecord'''
    examples = []
    for example in tf.python_io.tf_record_iterator(tfrecord_file):
        examples.append(tf.train.Example.FromString(example))
    return examples


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


if __name__ == "__main__":
    tfrecords = list_files('./mix1_ready/', '.tfrecords')
    
    for tfrecord in tfrecords:
        if '_train_synth_NPR_' in tfrecord:
            examples = read_examples(tfrecord)
            examples = correct_examples(examples)
            head_tail = os.path.split(tfrecord)
            new_name = os.path.join(head_tail[0], head_tail[1][:-10] +'_mod.tfrecords')
            print(tfrecord + " -> "+ new_name)
            write_tfrecord(new_name, examples)
            os.remove(tfrecord)
            