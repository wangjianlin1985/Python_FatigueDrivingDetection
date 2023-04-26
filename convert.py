from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import tensorflow as tf
import random

tf.app.flags.DEFINE_string(
    'dataset_dir', 'data',
    'The directory where the output TFRecords and temporary files are saved.')

FLAGS = tf.app.flags.FLAGS

_IMAGE_SIZE = None #Note!-not use,just show
_NUM_CHANNELS = None #Note!-not use,just show

# The number of images in the training set. Note!-not use,just show
_NUM_TRAIN_SAMPLES = None 

# The number of images to be kept from the training set for the validation set.
_NUM_VALIDATION = 0

# The number of images in the test set.  Note!-not use,just show
_NUM_TEST_SAMPLES = None

# Seed for repeatability.
_RANDOM_SEED = 0

class ImageReader(object):

    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes none-RGB jpeg data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        # You can use "tf.image.decode_jpeg" instead.
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=1)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(
            self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 1
        return image
    
def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

    Args:
        dataset_dir: The directory where the temporary files are stored.
        split_name: The name of the train/test split.

    Returns:
        An absolute file path.
    """
    return '%s/FACE_%s.tfrecord' % (dataset_dir, split_name)

def _get_filenames(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
        dataset_dir: A directory containing a set jpeg encoded images.

    Returns:
        A list of image file paths, relative to `dataset_dir`.
    """
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        photo_filenames.append(filename)
    return photo_filenames

def _extract_labels(label_filename):

    """Extract the labels into a dict of filenames to int labels.

    Args:
        labels_filename: The filename of the labels.

    Returns:
        A dictionary of filenames to int labels.
    """
    print('Extracting labels from: ', label_filename)
    label_file = tf.gfile.FastGFile(label_filename, 'r').readlines()
    label_lines = [line.rstrip('\n').split() for line in label_file]
    labels = {}
    for line in label_lines:
        assert len(line) == 2
        labels[line[0]] = int(line[1])
    return labels

def _int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
        values: A scalar or list of values.

    Returns:
        a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
        values: A string.

    Returns:
        a TF-Feature.
    """   
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_data),
        'image/format': _bytes_feature(image_format),
        'image/class/label': _int64_feature(class_id),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
    }))

def _convert_dataset(split_name, filenames, filename_to_class_id, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

      Args:
        split_name: The name of the dataset, either 'train' or 'valid'.
        filenames: A list of absolute paths to jpeg images.
        filename_to_class_id: A dictionary from filenames (strings) to class ids
          (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    print('Converting the {} split.'.format(split_name))
    # Train and validation splits are both in the train directory.
    if split_name in ['train', 'valid']:
        jpeg_directory = os.path.join(dataset_dir,'train')
    elif split_name == 'test':
        jpeg_directory = os.path.join(dataset_dir, 'test')

    with tf.Graph().as_default():
        image_reader = ImageReader()
        
        with tf.Session('') as sess:
            output_filename = _get_output_filename(dataset_dir, split_name)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for filename in filenames:
            # Read the filename:
                image_data = tf.gfile.FastGFile(
                    os.path.join(jpeg_directory, filename), 'rb').read()
                #print (len(image_data))
                height, width = image_reader.read_image_dims(sess, image_data)

                class_id = filename_to_class_id[filename]

                example = image_to_tfexample(image_data, 'jpeg'.encode(), height,
                                                     width, class_id)
                tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()
                

def run(dataset_dir):
    """Runs conversion operation.

    Args:
        dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    train_filename = _get_output_filename(dataset_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, 'test')

    if tf.gfile.Exists(train_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    train_validation_filenames = _get_filenames(
        os.path.join(dataset_dir, 'train'))
    test_filenames = _get_filenames(
        os.path.join(dataset_dir, 'test'))

    # Divide into train and validation:
    random.seed(_RANDOM_SEED)
    random.shuffle(train_validation_filenames)
    train_filenames = train_validation_filenames[_NUM_VALIDATION:]
    validation_filenames = train_validation_filenames[:_NUM_VALIDATION]

    train_validation_filenames_to_class_ids = _extract_labels(
        os.path.join(dataset_dir, 'train_labels.txt'))
    test_filenames_to_class_ids = _extract_labels(
        os.path.join(dataset_dir, 'test_labels.txt'))

    # Convert the train, validation, and test sets.
    _convert_dataset('train', train_filenames,
                   train_validation_filenames_to_class_ids, dataset_dir)
    _convert_dataset('valid', validation_filenames,
                   train_validation_filenames_to_class_ids, dataset_dir)
    _convert_dataset('test', test_filenames, test_filenames_to_class_ids,
                   dataset_dir)

    print('\nFinished converting the dataset!')

    
def main(_):
    assert FLAGS.dataset_dir
    run(FLAGS.dataset_dir)


