#!/usr/bin/env python3

'''
    This script is aimed at helping with the preparation of RGB masks for use to train computer vision models
    for semantic segmentation.
    The RGB masks that label each class in a picture of interest by a specified color are converted into
    a single channel (more precisely, 3 identical channels) mask and saved in a given folder. 
    
    Thank you to all the people who shared pieces of codes which inspired this script. Feel free to use, 
    share, and modify as you wish :)

    Usage: python3 rgbmask_to_singlechannelmask.py
'''

import os
import cv2
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import pathlib
import pandas as pd
import warnings

def loadColormapFromCSV(inputfolder):
    '''
        This function loads a csv file that contains the color (r, g, b)
        represeting each class in the in the RGB mask.

        inputfolder = Path to the csv file
    '''
    class_names_read = pd.read_csv(inputfolder)
    colormap = np.array(class_names_read.iloc[:,1:])

    return colormap

def maskOneHotEncoding(mask, classcolorfilefolder):
    '''
        This function converts the RGB mask into a list of length equal the number 
        of classes and the i-th elemnt of the list is a 0-1 matrix in which the pixels
        containing the i-th class in the RGB mask have the value 1 while the remaining
        pixels have value 0 

        mask = RGB mask image
        classcolorfilefolder = Folder in which the CSV file containing the class-color mapping is stored.
    '''
    colormap = loadColormapFromCSV(classcolorfilefolder)
    outputmask = [np.all(np.equal(mask, color),  axis=-1) for color in colormap]
    outputmask = np.stack(outputmask, axis=-1)

    return outputmask

def singleChannelMaskConversion(inputfolder, outputfolder, classcolorfilefolder):
    '''
        This function converts the one hot encoded mask into a single channel mask 
        (3 identical channels to be more precise), and saves the masks in the specified 
        output folder. If the folder does not exist, it will be created.

        inputfolder = Folder in which the RGB masks are stored.
        outputfolder = Folder in which the single channel masks will be stored.
        classcolorfilefolder = Folder in which the CSV file containing the class-color mapping is stored
    '''
    
    if not os.path.exists(outputfolder):
        warnings.warn("Folder not found, it will be created for you.")
        os.makedirs(outputfolder)

    rgbmaskfolder = pathlib.Path(inputfolder)
    rgbmaskpathlist = list(rgbmaskfolder.glob('**/*.png'))
    
    for rgbmaskpath in rgbmaskpathlist:        
        rgbmask = PIL.Image.open(rgbmaskpath)
        rgbmask = np.asarray(rgbmask)
        
        outputmask = maskOneHotEncoding(rgbmask, classcolorfilefolder)

        # Create array in which each entry (corresponding to image pixels) contains 
        # the class label number, e.g., 5 for the class 'car'
        singlechannelmask = np.argmax(outputmask, axis=-1)
        singlechannelmask = np.expand_dims(singlechannelmask, axis=-1)

        # Create mask as a 3 channel image for which all channels are identical
        mask = np.concatenate([singlechannelmask, singlechannelmask, singlechannelmask], axis=-1)
        
        # Pick rgbmask name (with extension, e.g., 'png')
        rgbmaskname = str(rgbmaskpath)[(len(inputfolder)+1):]
        output_path = os.path.join(outputfolder, rgbmaskname)

        # Save single channel mask to output folder with same name as rgb mask
        cv2.imwrite(output_path, mask)

def main():
    # Convert images to grayscale
    singleChannelMaskConversion(labelsdir, preppedlabelsdir, classcolorfilefolder)

if __name__ == '__main__':
    # Define input and output folder paths
    labelsdir = "/home/christian/Documents/Artifitial_Intelligence/MulticlassRGBMask_to_SingleChannelMask/dataset/train_labels"
    preppedlabelsdir = "/home/christian/Documents/Artifitial_Intelligence/MulticlassRGBMask_to_SingleChannelMask/dataset/train_labels_prepped"
    classcolorfilefolder = '/home/christian/Documents/Artifitial_Intelligence/MulticlassRGBMask_to_SingleChannelMask/dataset/class_dict.csv'

    # Conversion
    main()
    print(f"All files successfully converted to single chanel masks. Find them in: {preppedlabelsdir}")
