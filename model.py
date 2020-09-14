import os, time, math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from glob import glob
from ops import *
from config import Config

class DCGAN(object):
    def __init__(self, config):
        self.latent_dim = config.LATENT_DIM
        self.height = config.IMG_SIZE
        self.width = config.IMG_SIZE
        self.channel = 1
        #self.y_dim = 1
        self.latent_dm = config.LATENT_DIM
        self.batch_size = config.BATCH_SIZE
        self.global_step = tf.Variable(initial_value = 0, name='global_step', trainable=False, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                
        self.image = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel], name='real_images')
        #self.label = tf.placeholder(tf.float32, [None, self.y_dim], name='y')
        self.noise = tf.placeholder(tf.float32, [None, 1, 1, self.latent_dm], name='z')

        self.gene = self.generator(self.noise)
        self.real = self.discriminator(self.image)
        self.fake = self.discriminator(self.gene, reuse=True)

        # use Sigmoid, Please not confused
        self.loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real, labels=tf.ones_like(tf.nn.sigmoid(self.real))))
        self.loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake, labels=tf.zeros_like(tf.nn.sigmoid(self.fake))))

        ## Don't use Sigmoid, Please not confused
        #self.loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real, labels=tf.ones_like(self.real)))
        #self.loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake, labels=tf.zeros_like(self.fake)))

        self.loss_D = self.loss_D_real + self.loss_D_gene
        self.loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake, labels=tf.ones_like(tf.nn.sigmoid(self.fake))))

        self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        self.sample = self.generator(self.noise, is_training=False, reuse=True)

        print("Done building")

    def generator(self, input, is_training=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse) as scope:
            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}
            with slim.arg_scope([slim.conv2d_transpose],
                                kernel_size=[4, 4],
                                stride=[2, 2],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):

                # inputs : 1 x 1x100
                net = slim.conv2d_transpose(input, 256, scope='layer1') # 2, 2, 256
                net = slim.conv2d_transpose(net, 256, scope='layer2') # 4, 4, 256
                net = slim.conv2d_transpose(net, 128, scope='layer3') # 8, 8, 256
                net = slim.conv2d_transpose(net, 128, scope='layer4') # 16, 16, 128
                net = slim.conv2d_transpose(net, 64, scope='layer5') # 32, 32, 64
                net = slim.conv2d_transpose(net, 64, scope='layer6') # 64, 64, 64
                net = slim.conv2d_transpose(net, 32, scope='layer7') # 128, 128, 32
                net = slim.conv2d_transpose(net, 1, normalizer_fn=None, activation_fn=tf.nn.tanh, scope='layer8') # 256, 256, 1

                return net

    def discriminator(self, input, is_training=True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}
            with slim.arg_scope([slim.conv2d],
                                kernel_size = [4, 4],
                                stride = [2, 2],
                                activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):

                net = slim.conv2d(input, 32, normalizer_fn=None, scope='layer1')
                net = slim.conv2d(net, 64, scope='layer2')
                net = slim.conv2d(net, 64, scope='layer3')
                net = slim.conv2d(net, 128, scope='layer4')
                net = slim.conv2d(net, 128, scope='layer5')
                net = slim.conv2d(net, 256, scope='layer6')
                net = slim.conv2d(net, 256, scope='layer7')
                net = slim.conv2d(net, 1, kernel_size=[2, 2], stride=[1, 1], padding='VALID', normalizer_fn=None, activation_fn=None, scope='layer8')
                logits = tf.squeeze(net, axis=[1, 2])

                return logits

    def feature_match_layer(self, input, is_training=True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}
            with slim.arg_scope([slim.conv2d],
                                kernel_size = [4, 4],
                                stride = [2, 2],
                                activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):
                
                net = slim.conv2d(input, 32, normalizer_fn=None, scope='layer1')
                net = slim.conv2d(net, 64, scope='layer2')
                net = slim.conv2d(net, 64, scope='layer3')
                net = slim.conv2d(net, 128, scope='layer4')
                net = slim.conv2d(net, 128, scope='layer5')
                net = slim.conv2d(net, 256, scope='layer6')
                net = slim.conv2d(net, 256, scope='layer7')
                return net

    def anomaly_detector(self, para=0.1, method='feature'):
        
        self.test_images = tf.placeholder(tf.float32, [1, self.height, self.width, self.channel], name='test_images')
        self.ano_z = tf.get_variable('ano_z', shape = [1, 1, 1, self.latent_dim], dtype = tf.float32,
                                     initializer = tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32))
                
        self.ano_sample = self.generator(self.ano_z, is_training=False, reuse=True)

        self.res_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.test_images - self.ano_sample)))
        #self.res_loss = tf.reduce_mean(tf.squared_difference(self.test_images, self.ano_sample))

        if method is 'feature':
            i = self.feature_match_layer(self.test_images, is_training=False, reuse=True)
            z = self.feature_match_layer(self.ano_sample, is_training=False, reuse=True)            
            self.dis_loss = tf.reduce_mean(tf.reduce_sum(i-z))
            #self.dis_loss = tf.reduce_mean(tf.squared_difference(i, z))
        else:
            test_z = self.discriminator(self.test_images, is_training=False,  reuse=True)
            self.dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=test_z, labels=tf.ones_like(tf.nn.sigmoid(test_z))))

        self.anomaly_score = (1.-para)*self.res_loss + para*self.dis_loss

        t_vars = tf.trainable_variables()
        self.z_vars = [var for var in t_vars if 'ano_z' in var.name]


if __name__ == "__main__":
    model = DCGAN(Config())