# MulticlassRGBMask_to_SingleChannelMask
This script is aimed at helping with the preparation of RGB masks for use to train computer vision models
for semantic segmentation.

The RGB masks that label each class in a picture of interest by a specified color are converted into
a single channel (more precisely, 3 identical channels) mask and saved in a given folder. 

We thank all the individuals who shared pieces of codes which inspired this script. Feel free to use, 
share, and modify as you wish :)

Usage: python3 rgbmask_to_singlechannelmask.py
