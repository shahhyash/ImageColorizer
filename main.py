import numpy as np
from os import listdir
from os.path import join
import cv2

BW_DIR = 'bw'
COLOR_DIR = 'color'

_file = 'art294.jpg'

color_image = cv2.imread(join(COLOR_DIR, _file))
bw_image = cv2.imread(join(BW_DIR, _file), cv2.IMREAD_GRAYSCALE)    # Removing the flag cv2.IMREAD_GRAYSCALE causes it to read "RGB" values - which is essentially the same values three times

print("color_img rgb values @ (0,0):\t", color_image[0][0])
print("bw_img values at (0,0):\t\t", bw_image[0][0])