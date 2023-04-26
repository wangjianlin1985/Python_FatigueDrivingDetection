import tensorflow as tf
import model
import numpy as np
from PIL import Image

x = tf.placeholder(tf.float32, [None, 64*64])

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

num_classes = 2

checkpoint_dir = "path/to/model/"

sess = tf.Session()

logit = model.create_model(x, num_classes, keep_prob)
check_tag = tf.argmax(logit, 1)

saver = tf.train.Saver()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

def pre_process(img_path):
    img = Image.open(img_path).convert('L')
    img.save(img_path)
    
def check_state(img_path):
    global logit
    pre_process(img_path)
    image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
    split_name = img_path.split(".")[1]
    if split_name == "jpg":
        img_data = tf.image.decode_jpeg(image_raw_data)
    elif split_name == "png":
        img_data = tf.image.decode_png(image_raw_data)
    image = tf.image.convert_image_dtype(img_data, tf.float32)
    image = tf.image.resize_images(image, (64, 64), method=0)
    #image = img.astype(np.float32)
    image -= 0.5
    image *= 2
    image = tf.reshape(image, shape=(1,64*64))
    image = sess.run(image)
    output = sess.run(check_tag, {x:image, keep_prob:1.})
    print (output)
    if (output[0] == 1):
        print ("smile")
    else:      
        print ("no smile")
check_state("3.png")
