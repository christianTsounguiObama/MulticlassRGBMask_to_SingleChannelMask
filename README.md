# MulticlassRGBMask_to_SingleChannelMask
This script is aimed at helping with the preparation of RGB masks for use to train computer vision models
for semantic segmentation.

The RGB masks that label each class in a picture of interest by a specified color are converted into
a single channel (more precisely, 3 identical channels) mask and saved in a given folder. 

The code uses class-color mapping defined in a csv file, but can be easily adapted to accomodate mapping
provided in other formats, or provided in code as an array.

![illustration](https://github.com/christianTsounguiObama/MulticlassRGBMask_to_SingleChannelMask/assets/104940386/a6ca4c45-6d7c-4bb1-8a18-d9b181b2ecf7)

## Usage
Once the paths for the rgb masks folder, output folder, and class-color mapping file are defined in the script,
run the following command to perform the conversion:
```bash
python3 rgbmask_to_singlechannelmask.py
```


## Python version and packages
The code was written and tested with Python 3.10.12. The following packages might need to installed:
- cv2
- pillow
- numpy
- pathlib
- pandas

## Example dataset
To test the code, we provide you with a dataset of three pictures extracted from the CAMVID dataset downloaded at 
https://www.kaggle.com/datasets/carlolepelaars/camvid. The 'train' folder contains the groud truth images, 
the 'train_labels' folder contains the labeled RGB masks of each of the ground truth images, and the 'train_labels_prepped' 
contains the converted single channel masks.

## Acknowledgment
Thank you to all the people who shared pieces of codes which inspired this script. Feel free to use, 
share, and modify as you wish.

Any feedback will be highly appreciated :)
