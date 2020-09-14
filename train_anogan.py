import utils, cv2
import os, time, math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import Config
from model import DCGAN
from utils import read_images

config = Config()
model = DCGAN(config)
config.BATCH_SIZE = 1

t_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(t_vars, print_info=True)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
    train_D = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.loss_D, var_list=model.vars_D)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')):
    train_G = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.loss_G, global_step=model.global_step, var_list=model.vars_G)


sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

images, labels = read_images(config.PATH_DATA, "folder")
num_iters = len(images) // config.BATCH_SIZE

cnt = 0
length = 5
sample_noise = np.random.uniform(-1., 1., size=[length*length, 1, 1, config.LATENT_DIM])

img_path = "temp"
if not os.path.isdir("temp"):
    os.mkdir("temp")

with tf.Session(config=sess_config) as sess:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    #summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    sess.run(init)

    model_checkpoint_name = config.PATH_CHECKPOINT + "/model.ckpt"    
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

    model.anomaly_detector()

    #lr = tf.train.exponential_decay(0.01, model.global_step, decay_steps=8000, decay_rate=0.9)
    
    #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='anomaly_detector')):
        #train_Z = tf.train.AdamOptimizer(learning_rate=lr).minimize(model.anomaly_score, global_step=model.global_step, var_list=model.z_vars)
    train_Z = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.anomaly_score, var_list=model.z_vars)

    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
    print([str(i.name) for i in not_initialized_vars])
    sess.run(tf.variables_initializer(not_initialized_vars))

    for epoch in range(config.EPOCH):
        for idx in range(num_iters):
            st = time.time()
            image_batch = images[idx*config.BATCH_SIZE:(idx+1)*config.BATCH_SIZE]
            _, score, loss_Z, global_step, ano_z = sess.run([train_Z, model.anomaly_score, model.res_loss, model.global_step, model.ano_z], feed_dict={model.test_images:image_batch})

            cnt = cnt + config.BATCH_SIZE
            if cnt % 20 == 0:
                string_print = "Epoch = %d Count = %d Score = %.4f Current_Loss_Z = %.4f Time = %.2f"%(epoch, cnt, score, loss_Z, time.time()-st)
                utils.LOG(string_print)
                st = time.time()

            #if global_step%200 == 0 or global_step is 0:
            if idx is 0:
                print("Performing validation")
                results=None
                samples = sess.run(model.ano_sample)
                #samples = (samples+1)/2.0
                sample = (samples[0]+1)*127.5
                #error = samples - (image_batch+1)*127.5
                image = (image_batch[0]+1)*127.5
                error = cv2.absdiff(sample, image)
                error = cv2.threshold(error, 20, 255, cv2.THRESH_BINARY)[1]
                cnts, _ = cv2.findContours(np.array(error, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                for c in cnts:
                    cv2.fillPoly(color_image, [c], (0, 0, 255))
                    #cv2.drawContours(color_image, [c], -1, (0, 0, 255))
                cv2.imwrite("temp/" + str(epoch) + "_color_image.png", color_image)
                cv2.imwrite("temp/" + str(epoch) + "_image.png", image)
                cv2.imwrite("temp/" + str(epoch) + "_sample.png", sample)
                cv2.imwrite("temp/" + str(epoch) + "_error.png", error)

                #if results is None:
                #    results = error
                #else:
                #    results = np.vstack((results, error))
                #utils.print_sample_data(results, config.BATCH_SIZE, "temp/" + str(global_step) + "_gene_data.png")

        images, labels = utils.data_shuffle(images, labels)
        cnt = 0

        if epoch % config.CHECKPOINTS_STEP == 0:
            # Create directories if needed
            if not os.path.isdir("%s/%04d"%("checkpoints_anogan",epoch)):
                os.makedirs("%s/%04d"%("checkpoints_anogan",epoch))

            print('Saving model with global step %d ( = %d epochs) to disk' % (global_step, epoch))
            saver.save(sess, "%s/%04d/model.ckpt"%("checkpoints_anogan",epoch))

        # Save latest checkpoint to same file name
        print('Saving model with %d epochs to disk' % (epoch))
        saver.save(sess, config.PATH_ANOGAN_CHECKPOINT + "/model.ckpt")

    
    #results=None
    #for idx in range(length):
    #    X = sess.run(model.sample, feed_dict={model.noise:sample_noise[length*idx:length*(idx+1)]})
    #    X = (X+1)/2.0
    #    if results is None:
    #        results = X
    #    else:
    #        results = np.vstack((results, X))
    #utils.save_plot_generated(results, length, "gene_data.png")