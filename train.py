#import utils, cv2
#import os, time, math
#import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
##from glob import glob
##from ops import *
#from config import Config
#from model import DCGAN
#from utils import read_images


#config = Config()

#model = DCGAN(config)
#model.anomaly_detector()
#t_vars = tf.trainable_variables()
#slim.model_analyzer.analyze_vars(t_vars, print_info=True)

#with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
#    train_D = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.loss_D, var_list=model.vars_D)
#with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')):
#    train_G = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.loss_G, global_step=model.global_step, var_list=model.vars_G)

#with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
#    train_Z = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.anomaly_score, global_step=model.global_step, var_list=model.z_vars)

#sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

#images, labels = read_images("medical/category2", "folder")
#num_iters = len(images) // config.BATCH_SIZE

#length = 5
#sample_noise = np.random.uniform(-1., 1., size=[length*length, 100])
#sample_label = [[0] for i in range(length*length)]

#with tf.Session(config=sess_config) as sess:
#    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
#    #summary_op = tf.summary.merge_all()
#    init = tf.global_variables_initializer()
#    sess.run(init)

#    model_checkpoint_name = config.PATH_CHECKPOINT + "/model.ckpt"
#    if config.IS_CONTINUE:
#        print('Loaded latest model checkpoint---------------------------------------')
#        saver.restore(sess, model_checkpoint_name)

#    cnt=0

#    #for epoch in range(config.EPOCH):
#    #    for idx in range(num_iters):
#    #        st = time.time()
#    #        image_batch = images[idx*config.BATCH_SIZE:(idx+1)*config.BATCH_SIZE]
#    #        label_batch = labels[idx*config.BATCH_SIZE:(idx+1)*config.BATCH_SIZE]
#    #        noise_batch = np.random.uniform(-1., 1., size=[config.BATCH_SIZE, 100])

#    #        _, loss_D = sess.run([train_D, model.loss_D], feed_dict={model.image:image_batch, model.noise:noise_batch, model.label:label_batch})
#    #        _, loss_G = sess.run([train_G, model.loss_G], feed_dict={model.image:image_batch, model.noise:noise_batch, model.label:label_batch})
#    #        _, loss_G, global_step = sess.run([train_G, model.loss_G, model.global_step], feed_dict={model.image:image_batch, model.noise:noise_batch, model.label:label_batch})

#    #        cnt = cnt + config.BATCH_SIZE
#    #        if cnt % 20 == 0:
#    #            string_print = "Epoch = %d Count = %d Current_Loss_D = %.4f Current_Loss_G = %.4f Time = %.2f"%(epoch, cnt, loss_D, loss_G, time.time()-st)
#    #            utils.LOG(string_print)
#    #            st = time.time()

#    #        if global_step%config.PRINT_STEP == 0 or global_step is 0:
#    #            print("Performing validation")
#    #            results=None
#    #            for idx in range(length):
#    #                X = sess.run(model.sample, feed_dict={model.noise:sample_noise[length*idx:length*idx+length], model.label:sample_label[length*idx:length*idx+length]})
#    #                X = (X+1)/2.0
#    #                if results is None:
#    #                    results = X
#    #                else:
#    #                    results = np.vstack((results, X))
#    #            utils.save_plot_generated(results, length, "sample_data/" + str(global_step) + "_" + str(epoch) + "_gene_data.png")

#    #    image, labels = utils.data_shuffle(images, labels)
#    #    cnt = 0

#    #    if epoch % config.CHECKPOINTS_STEP == 0:
#    #        # Create directories if needed
#    #        if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
#    #            os.makedirs("%s/%04d"%("checkpoints",epoch))

#    #        print('Saving model with global step %d ( = %d epochs) to disk' % (global_step, epoch))
#    #        saver.save(sess, "%s/%04d/model.ckpt"%("checkpoints",epoch))

#    #    # Save latest checkpoint to same file name
#    #    print('Saving model with %d epochs to disk' % (epoch))
#    #    saver.save(sess, model_checkpoint_name)



#    #---------------------------------------------
#    for epoch in range(config.EPOCH*5):
#        for idx in range(num_iters):
#            st = time.time()
#            image_batch = images[idx*config.BATCH_SIZE:(idx+1)*config.BATCH_SIZE]
#            label_batch = labels[idx*config.BATCH_SIZE:(idx+1)*config.BATCH_SIZE]
#            _, score, loss_Z, global_step= sess.run([train_Z, model.anomaly_score, model.res_loss, model.global_step], feed_dict={model.test_images:image_batch, model.ano_z_label:label_batch})

#            cnt = cnt + config.BATCH_SIZE
#            if cnt % 20 == 0:
#                string_print = "Epoch = %d Count = %d Score = %.4f Current_Loss_Z = %.4f Time = %.2f"%(epoch, cnt, score, loss_Z, time.time()-st)
#                utils.LOG(string_print)
#                st = time.time()

#            if global_step%200 == 0 or global_step is 0:
#                print("Performing validation")
#                results=None
#                samples = sess.run(model.ano_sample, feed_dict={model.ano_z_label:label_batch})
#                samples = (samples+1)/2.0
#                error = samples - image_batch
#                if results is None:
#                    results = error
#                else:
#                    results = np.vstack((results, error))
#                utils.print_sample_data(results, config.BATCH_SIZE, "temp/" + str(global_step) + "_gene_data.png")

#        image, labels = utils.data_shuffle(images, labels)
#        cnt = 0

#        if epoch % config.CHECKPOINTS_STEP == 0:
#            # Create directories if needed
#            if not os.path.isdir("%s/%04d"%("checkpoints_anogan",epoch)):
#                os.makedirs("%s/%04d"%("checkpoints_anogan",epoch))

#            print('Saving model with global step %d ( = %d epochs) to disk' % (global_step, epoch))
#            saver.save(sess, "%s/%04d/model.ckpt"%("checkpoints_anogan",epoch))

#        # Save latest checkpoint to same file name
#        print('Saving model with %d epochs to disk' % (epoch))
#        saver.save(sess, config.PATH_ANOGAN_CHECKPOINT + "/model.ckpt")


import utils, cv2
import os, time, math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from glob import glob
#from ops import *
from config import Config
from model import DCGAN
from utils import read_images


config = Config()

model = DCGAN(config)

t_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(t_vars, print_info=True)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
    train_D = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.loss_D, var_list=model.vars_D)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')):
    train_G = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.loss_G, global_step=model.global_step, var_list=model.vars_G)

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

images, labels = read_images("data", "folder")
num_iters = len(images) // config.BATCH_SIZE

cnt = 0
length = 5
sample_noise = np.random.uniform(-1., 1., size=[length*length, 1, 1, config.LATENT_DIM])

with tf.Session(config=sess_config) as sess:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    #summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    sess.run(init)

    model_checkpoint_name = config.PATH_CHECKPOINT + "/model.ckpt"
    if config.IS_CONTINUE:
        print('Loaded latest model checkpoint---------------------------------------')
        saver.restore(sess, model_checkpoint_name)

    for epoch in range(config.EPOCH):
        for idx in range(num_iters):
            st = time.time()
            image_batch = images[idx*config.BATCH_SIZE:(idx+1)*config.BATCH_SIZE]
            noise_batch = np.random.uniform(-1., 1., size=[config.BATCH_SIZE, 1, 1, config.LATENT_DIM])

            for _ in range(1):
                _, loss_D = sess.run([train_D, model.loss_D], feed_dict={model.image:image_batch, model.noise:noise_batch})
            #_, loss_G = sess.run([train_G, model.loss_G], feed_dict={model.image:image_batch, model.noise:noise_batch})
            _, loss_G, global_step = sess.run([train_G, model.loss_G, model.global_step], feed_dict={model.image:image_batch, model.noise:noise_batch})

            cnt = cnt + config.BATCH_SIZE
            if cnt % 20 == 0:
                string_print = "Epoch = %d Count = %d Current_Loss_D = %.4f Current_Loss_G = %.4f Time = %.2f"%(epoch, cnt, loss_D, loss_G, time.time()-st)
                utils.LOG(string_print)
                st = time.time()

            #if global_step%config.PRINT_STEP == 0 or global_step is 0:
            if idx is num_iters-1 and idx%2 == 0:
                print("Performing validation")
                results=None
                for idx in range(length):
                    X = sess.run(model.sample, feed_dict={model.noise:sample_noise[length*idx:length*(idx+1)]})
                    X = (X+1)/2.0
                    if results is None:
                        results = X
                    else:
                        results = np.vstack((results, X))
                utils.save_plot_generated(results, length, "sample_data/" + str(global_step) + "_" + str(epoch) + "_gene_data.png")

        images, labels = utils.data_shuffle(images, labels)
        cnt = 0

        if epoch % config.CHECKPOINTS_STEP == 0:
            # Create directories if needed
            if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
                os.makedirs("%s/%04d"%("checkpoints",epoch))

            print('Saving model with global step %d ( = %d epochs) to disk' % (global_step, epoch))
            saver.save(sess, "%s/%04d/model.ckpt"%("checkpoints",epoch))

        # Save latest checkpoint to same file name
        print('Saving model with %d epochs to disk' % (epoch))
        saver.save(sess, model_checkpoint_name)