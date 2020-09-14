# Anomaly Detection with GAN
The repository has implemented the **Anomaly Detection with GAN** and has been applied to the **Flickr Face HQ Dataset.**   

**Requirements**
* python 3.6   
* tensorflow-gpu==1.14   
* pillow
* matplotlib


## Concept
**Architecture**


I made the kernel size different from the picture above when i implemented it.

## Files and Directories
* config.py : A file that stores various parameters and path settings.
* model.py : ANOGAN's network implementation file
* train.py : This file load the data and learning with GAN.
* train_anogan.py : This file load the GAN model and then learns the detector.
* utils.py : Various functions such as loading data* 

## Train Flickr Face HQ Dataset
1. Download [Flickr Face HQ Dataset](https://github.com/NVlabs/ffhq-dataset)
2. The **read_images** function on **utils.py** has a subfolder that has an image corresponding to each class in the root folder.   
   ```
   ROOT_FOLDER
      |   
      |--------SUBFOLDER (Class 0)   
      |          |------image1.jpg   
      |          |------image2.jpg   
      |          |------etc..   
      |--------SUBFOLDER (Class 1)   
      |          |------image1.jpg   
      |          |------image2.jpg   
      |          |------etc..
   ```
      
   Please create a folder to store learning images and insert learning images to this standard. I used 10,000 images in thumbnails128x128 folder. (from 00000 folder to 09000 folder)
   The path i used is as follows.
   ```
   Example: Copy thumbnails128x128 to D:/data and Write D:/data/thumbnails128x128 on config.py
   ```
3. Run **train.py**
4. Run **train_anogan.py**
5. The result images are stored per epoch in the **temp**

## Future work
* Upload result images

## Reference
* Schlegl, Thomas, et al. "Unsupervised anomaly detection with generative adversarial networks to guide marker discovery." International conference on information processing in medical imaging. Springer, Cham, 2017.
* https://github.com/tkwoo/anogan-keras
