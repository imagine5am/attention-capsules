import argparse
import cv2
import numpy as np
import os
import tensorflow as tf
import traceback


def read_examples(tfrecord_file):
    '''Returns a list of examples read from a single tfrecord'''
    examples = []
    try:
        for example in tf.python_io.tf_record_iterator(tfrecord_file):
            examples.append(tf.train.Example.FromString(example))
    except Exception:
        print('Error in file: ' + tfrecord_file)
        traceback.print_exc()
    return examples


def slideshow(images):
    '''
    Takes in a list of tuples.
    0th value of tuple is an RGB image.
    1st value of tuple is the label.
    '''
    num_img = len(images)
    current = 0
    timeout = 2000
    window_size = (1280, 720)
    
    print('Number of Examples:', num_img)
    print('Press N for next and P for previous.')
    print('Any other key to exit.')
    finish = False
    while not finish:
        title = images[current][1]
        print('Current Image:', current, 'Label:', title)
        
        image = cv2.cvtColor(images[current][0], cv2.COLOR_RGB2BGR)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, window_size[0], window_size[1])
        cv2.imshow(title, image)
        key = cv2.waitKey(timeout)

        if key == ord('n') or key == -1:
            if current == num_img - 1:
                finish = True
            else:
                current = (current + 1) % num_img
        elif key == ord('p'):
            if current == 0:
                current = num_img - 1
            else:
                current -= 1
        else:
            finish = True
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='View images in a tfrecord')
    parser.add_argument('--tfrecord', help='Location of tfrecord')
    args = parser.parse_args()
    
    tfrecord = args.tfrecord
    examples = read_examples(tfrecord)
    
    images = []
    for example in examples:
        label_str = example.features.feature['image/text'].bytes_list.value[0]
        label = label_str.decode('utf-8')

        img_str = example.features.feature['image/encoded'].bytes_list.value[0]
        img_np_arr = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(img_np_arr, 1)
        
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # images.append((im_rgb[:256, :480], label))
        images.append((im_rgb, label))
        
    slideshow(images)
