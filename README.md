# MulticlassRGBMask_to_SingleChannelMask
This script is aimed at helping with the preparation of RGB masks for use to train computer vision models
for semantic segmentation.

The RGB masks that label each class in a picture of interest by a specified color are converted into
a single channel (more precisely, 3 identical channels) mask and saved in a given folder. 

Thank you to all the people who shared pieces of codes which inspired this script. Feel free to use, 
share, and modify as you wish :)

Usage: python3 rgbmask_to_singlechannelmask.py

# Example dataset
To test the code, we provide you with a dataset of three pictures extracted from the CAMVID dataset downloaded at 
https://www.kaggle.com/datasets/carlolepelaars/camvid. The 'train' folder contains the groud truth images, 
the 'train_labels' folder contains the labeled RGB masks of each of the ground truth images, and the 'train_labels_prepped' 
contains the converted single channel masks.


