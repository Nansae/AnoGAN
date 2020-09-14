import os
import cv2
import numpy as np
import datetime, time
import matplotlib.pyplot as plt

def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)

def read_images(dataset_path, mode):
    imagepaths, labels = list(), list()
    if mode == 'file':
        # Read dataset file
        with open(dataset_path) as f:
            data = f.read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
        # List the directory
        try:  # Python 2
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:  # Python 3
            classes = sorted(os.walk(dataset_path).__next__()[1])
        # List each sub-directory (the classes)
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:  # Python 2
                walk = os.walk(c_dir).next()
            except Exception:  # Python 3
                walk = os.walk(c_dir).__next__()
            # Add each image to the training set
            for sample in walk[2]:
                # Only keeps jpeg images
                if sample.endswith('.jpg') or sample.endswith('.jpeg') or sample.endswith('.bmp') or sample.endswith('.png'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append([label])
            label += 1
    else:
        raise Exception("Unknown mode.")

    images = list()
    for p in imagepaths:
        image = cv2.imread(p, 0)
        if image is  None:
            continue
        image = cv2.resize(image, (256, 256))
        image = np.float32(image)/127.5 - 1
        image = np.expand_dims(image, axis=2)    
        images.append(image)
        #cv2.imshow("image", image)

    images = np.array(images)
    labels = np.array(labels, dtype=np.int32)

    print("images: %d" % len(images))
    print("labels: %d" % len(labels))

    if len(images) != len(labels):
        print("Error---------------------")

    else:
        return images, labels


def data_shuffle(images, labels):
    s = np.arange(images.shape[0])
    np.random.shuffle(s)
    return images[s], labels[s]

def plot_generated(sample_data, n):
    plt.figure(figsize=(10, 10))
    for i in range(n * n):
        plt.subplot(n, n, 1+i)
        plt.axis('off')
        plt.imshow((sample_data[i]).reshape([256, 256]), cmap='gray')
    plt.savefig('fig.png', dpi=300)
    plt.show()

def save_plot_generated(sample_data, n, str):
    plt.figure(figsize=(10, 10))
    for i in range(n * n):
        plt.subplot(n, n, 1+i)
        plt.axis('off')
        plt.imshow((sample_data[i]).reshape([256, 256]), cmap='gray')
    plt.savefig(str, dpi=300)
    
def save_linear_plot_generated(sample_data, n, str):
    plt.figure(figsize=(n, 1))
    for i in range(n):
        plt.subplot(n, 1, 1+i)
        plt.axis('off')
        plt.imshow((sample_data[i]).reshape([256, 256]), cmap='gray')
    plt.savefig(str, dpi=300)

def print_sample_data(sample_data, max_print=10, str=""):
    print_images = sample_data[:max_print,:]
    print_images = print_images.reshape([max_print, 256, 256])
    print_images = print_images.swapaxes(0, 1)
    print_images = print_images.reshape([256, max_print * 256])
  
    plt.figure(figsize=(max_print, 1))
    plt.axis('off')
    plt.imshow(print_images, cmap='gray')
    #plt.show()
    plt.savefig(str, dpi=300)