import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

image = cv2.imread('test1.jpg')

input_data = tf.placeholder(tf.float32, [None, image.shape[0], image.shape[1], 3])

weigths = {
    'w1': tf.Variable(tf.orthogonal_initializer().__call__(shape=[5, 5, 3, 6])),
    'w2': tf.Variable(tf.orthogonal_initializer().__call__(shape=[3, 3, 6, 12])),
    'w3': tf.Variable(tf.orthogonal_initializer().__call__(shape=[3, 3, 12, 3]))
}
# x = tf.reshape(input_data, [-1, image.shape[0], image.shape[1], 3])

conv1 = tf.nn.relu(tf.nn.conv2d(input_data, weigths['w1'], strides=[1, 2, 2, 1], padding='SAME'))

conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weigths['w2'], strides=[1, 1, 1, 1], padding='SAME'))

mp = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv3 = tf.nn.relu(tf.nn.conv2d(mp, weigths['w3'], strides=[1, 1, 1, 1], padding='SAME'))

mp1 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    conv = sess.run(conv3, feed_dict={input_data: [image]})
    count = 0
    print(conv1.shape)
    d = np.squeeze(conv, 0)
    print(d.shape)
    cv2.imwrite('img.jpg', d)
sess.close()



