import tensorflow as tf
import os
import model
import data_provider

TEXT_EXAMPLES = 800
num_classes = 2

global_step = tf.Variable(0, trainable=False)

x = tf.placeholder(tf.float32, [None, 64*64])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

image, label = data_provider.get_data("test")
image = tf.reshape(image, shape=(1,64*64))
label = tf.reshape(label, shape=(1,2))

logit = model.create_model(x, num_classes, keep_prob)
correct_pred = tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))

correct_num = 0

saver = tf.train.Saver()

checkpoint_dir = "path/to/model/"

with tf.Session() as sess:
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    for i in range(TEXT_EXAMPLES):
        test_image, test_label = sess.run([image, label])
        correct = sess.run([correct_pred], feed_dict=
                           {x:test_image, y:test_label, keep_prob:1.})
        
        if correct[0] == True:
            #print ("*************", correct)
            correct_num += 1

    acc = correct_num / TEXT_EXAMPLES
    print ("After %s iterator(s) the accuracy is %f" %(global_step, acc))
    
