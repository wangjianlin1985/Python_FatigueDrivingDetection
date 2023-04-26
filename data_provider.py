import os
import tensorflow as tf

_FILE_PATTERN = 'FACE_%s.tfrecord'
dataset_dir = 'data'
reader = tf.TFRecordReader()


keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'image/class/label': tf.FixedLenFeature([1], tf.int64),
}

num_classes = 2

def get_data(split_name):
    file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)
    filename_queue = tf.train.string_input_producer([file_pattern])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features = keys_to_features )
    #image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.image.decode_png(features['image/encoded'])
    #label = tf.cast(features['image/class/label'],tf.float32)
    label = tf.one_hot(features['image/class/label'], num_classes)
    label = tf.reshape(label, shape=(num_classes,))
    print ("label:", label)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image -= 0.5
    image *= 2
    image = tf.reshape(image, shape=(64*64,))
    print (image, label)
    return (image, label)


#test_image, test_label = get_data("test")
#print (test_image)
#print (test_label)
