import cv2
from os import listdir
from os.path import isfile, join

color_pictures = [pic for pic in listdir("color")]

for pic in color_pictures:
    pic_path = join('color', pic)
    bw_path = join('bw', pic)

    pic_grayscale = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(bw_path, pic_grayscale)