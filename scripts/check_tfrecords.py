import argparse
import os
import tensorflow as tf
import traceback

from tqdm import tqdm


def get_img_and_label(example):
    '''returns the image and label feature in an Example'''
    label_str = example.features.feature['image/text'].bytes_list.value[0]
    label = label_str.decode('utf-8')
    
    img_str = example.features.feature['image/encoded'].bytes_list.value[0]
    img_np_arr = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(img_np_arr, 1)
    
    return img, label


def check_tfrecord(tfrecord_file):
    try:
        for example in tf.python_io.tf_record_iterator(tfrecord_file):
            img, label = get_img_and_label(example)
            char_ids_padded = example.features.feature['image/class'].bytes_list.value[0]
            char_ids_unpadded = example.features.feature['image/unpadded_class'].bytes_list.value[0]
            
            print(label)
            
    except Exception:
        print('Error in file: ' + tfrecord_file)
        traceback.print_exc()


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
    parser = argparse.ArgumentParser(description='Counts number of examples.')
    parser.add_argument('--tfrecords_loc', help='Location of tfrecords.')
    args = parser.parse_args()
    
    tfrecords_loc = args.tfrecords_loc
    tfrecords = list_files(src_loc=tfrecords_loc, file_extension='.tfrecords')
    count = 0
    
    for tfrecord in tqdm(tfrecords):
        check_tfrecord(tfrecord)
    