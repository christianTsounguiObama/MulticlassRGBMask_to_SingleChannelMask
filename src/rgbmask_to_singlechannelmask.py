import os
import cv2
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import pathlib
import pandas as pd

def prep_mask(mask, colormap):
    output_mask = []
    for i, color in enumerate(colormap):
        cmap = np.all(np.equal(mask, color),  axis=-1)
        output_mask.append(cmap)

    output_mask = np.stack(output_mask, axis=-1)
    return output_mask

def convert_to_grayscale_mask(input_folder, output_folder, colormap):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of image filenames in the input folder
    dir = pathlib.Path(input_folder)
    image_files = list(dir.glob('**/*.png'))
    
    for img_file in image_files:
        # Read the RGB mask using OpenCV        
        img = PIL.Image.open(img_file)
        img = np.asarray(img)
        
        # Convert to multiclass grayscale image
        output_mask = []
        for i, color in enumerate(colormap):
            cmap = np.all(np.equal(img, color),  axis=-1)
            output_mask.append(cmap)
        
        output_mask = np.stack(output_mask, axis=-1)
        grayscale_mask = np.argmax(output_mask, axis=-1)
        
        grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)
        mask = np.concatenate([grayscale_mask, grayscale_mask, grayscale_mask], axis=-1)
        img_path = str(img_file)[(len(input_folder)+1):]
        
        output_path = os.path.join(output_folder, img_path)
        cv2.imwrite(output_path, mask)

        print(f"Converted {img_file} to grayscale and saved as {output_path}")

# Input and output folder paths
input_folder = "/home/christian/Documents/Robotics/DNN/dataset/CamVid/test_labels"
output_folder = "/home/christian/Documents/Robotics/DNN/dataset/CamVid/test_labels_prepped"

#Loading class labels dataset
class_names_read = pd.read_csv('/home/christian/Documents/Robotics/DNN/dataset/CamVid/class_dict.csv')

# pixel labels in the video frames
class_names = np.array(class_names_read['name']).T
class_names = class_names.tolist()

# Labels colormap
colormap = np.array(class_names_read)[:,1:]

# Convert images to grayscale
convert_to_grayscale_mask(input_folder, output_folder, colormap)
