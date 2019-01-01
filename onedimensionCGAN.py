import numpy as np
import wfdb as wf
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import os, imageio
from tqdm import tqdm
import mitecg
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("datapath", help="directory to the mit ecg database")
args = parser.parse_args()



batch_size = 40
HEARTBEATSAMPLES = 100
LABEL = 5
z_dim = 100
ECG = mitecg.ReadMitEcg(args.datapath, 10000, [1, 2, 3, 4, 5], True, SCALEDSAMPLES = HEARTBEATSAMPLES)




X = tf.placeholder(dtype=tf.float32, shape=[None, HEARTBEATSAMPLES, 1], name='X')
y_label = tf.placeholder(dtype=tf.float32, shape=[None, HEARTBEATSAMPLES, LABEL], name='y_label')
noise = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='noise')
y_noise = tf.placeholder(dtype=tf.float32, shape=[None, LABEL], name='y_noise')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)


#discriminator part
def discriminator(heartbeat, label, reuse=None, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):
        h0 = tf.concat([heartbeat, label], axis=2)
        h0 = lrelu(tf.layers.conv1d(h0, kernel_size=5, filters=64, strides=2, padding='same'))
        
        h1 = tf.layers.conv1d(h0, kernel_size=5, filters=128, strides=2, padding='same')
        h1 = lrelu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))
        
        h2 = tf.layers.conv1d(h1, kernel_size=5, filters=256, strides=2, padding='same')
        h2 = lrelu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))
        
        h3 = tf.layers.conv1d(h2, kernel_size=5, filters=512, strides=2, padding='same')
        h3 = lrelu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))
        
        #h4 = tf.contrib.layers.flatten(h3)
        #h4 = tf.layers.dense(h4, units=1)
        h4 = tf.layers.dense(h3, units=1)
        return tf.nn.sigmoid(h4), h4



#generator part
def generator(z, label, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        d = 3
        z = tf.concat([z, label], axis=1)
        
        h0 = tf.layers.dense(z, 20, tf.nn.relu)
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))
        
        h1 = tf.layers.dense(h0, 50, tf.nn.relu)
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))
        
        
        
        h2 = tf.layers.dense(h1, HEARTBEATSAMPLES, tf.nn.tanh, name='g')

        h2 = tf.reshape(h2, shape=[-1, HEARTBEATSAMPLES, 1])
        return h2

#loss function
g = generator(noise, y_noise)
d_real, d_real_logits = discriminator(X, y_label)
d_fake, d_fake_logits = discriminator(g, y_label, reuse=True)


vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

loss_d_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_real_logits, tf.ones_like(d_real)))
loss_d_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.zeros_like(d_fake)))
loss_g = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.ones_like(d_fake)))
loss_d = loss_d_real + loss_d_fake


#optimization function
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state('net/')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored")
else:
    pass


samples = []
loss = {'d': [], 'g': []}
typeName = ["Normal beat", "Left bundle branch block beat", "Right bundle branch block beat", "Aberrated atrial premature beat", "Premature ventricular contraction"]

for i in range(5000):
    
    batch_xs, batch_ys = ECG.nextbatch(batch_size)
    batch_xs = np.reshape(batch_xs, (-1, HEARTBEATSAMPLES, 1))
    n = np.random.uniform(-1.0, 1.0, [batch_ys.shape[0], z_dim]).astype(np.float32)
    yn = np.copy(batch_ys)
    yl = np.reshape(batch_ys, [batch_ys.shape[0], 1, LABEL])
    yl = yl * np.ones([batch_ys.shape[0], HEARTBEATSAMPLES, LABEL])
    
    d_ls, g_ls = sess.run([loss_d, loss_g], feed_dict={X: batch_xs, noise: n, y_label: yl, y_noise: yn, is_training: True})
    loss['d'].append(d_ls)
    loss['g'].append(g_ls)
    
    sess.run(optimizer_d, feed_dict={X: batch_xs, noise: n, y_label: yl, y_noise: yn, is_training: True})
    sess.run(optimizer_g, feed_dict={X: batch_xs, noise: n, y_label: yl, y_noise: yn, is_training: True})
    sess.run(optimizer_g, feed_dict={X: batch_xs, noise: n, y_label: yl, y_noise: yn, is_training: True})
    
    
    print(i, d_ls, g_ls)
    #print(y_samples)
    
    if (i%100 == 0):
        z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
        y_samples = np.zeros([batch_size, LABEL])
        for k in range(LABEL):
            for j in range(LABEL):
                if (batch_size > k * LABEL + j):
                    y_samples[k * LABEL + j, j] = 1
        gen_heart = sess.run(g, feed_dict={noise: z_samples, y_noise: y_samples, is_training: False})
        print(gen_heart)
        print(gen_heart.shape)
        plt.figure()
        plt.subplot(511)
        plt.plot((gen_heart[0,:,:]).flatten())
        plt.title(typeName[0])

        plt.subplot(512)
        plt.plot((gen_heart[1,:,:]).flatten())
        plt.title(typeName[1])
    
        plt.subplot(513)
        plt.plot((gen_heart[2,:,:]).flatten())
        plt.title(typeName[2])

        plt.subplot(514)
        plt.plot((gen_heart[3,:,:]).flatten())
        plt.title(typeName[3])

        plt.subplot(515)
        plt.plot((gen_heart[4,:,:]).flatten())
        plt.title(typeName[4])
        plt.savefig("cgansamplesingle/" + str(i) + ".png")
        #plt.show()
    #print(gen_heart)
    #for databeat in batch_xs:
        #lines = plt.plot(databeat)#, TIME[indoneperiod], M[:, 1][indoneperiod])
        #plt.xlabel("TIME")
        #plt.title("ECG wave two chanels")
        #plt.show()
    #print(batch_xs.shape)
    #print(batch_ys.shape)
    #print(batch_ys)
    saver.save(sess, 'net/cgan.ckpt')


