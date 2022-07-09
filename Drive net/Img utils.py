import os
from os import listdir
import tifffile

import numpy
from PIL import Image, TiffImagePlugin

# load all images in a directory
from matplotlib import pyplot

img_dir_source = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "datasets", "Drive", "sources")
img_dir_save = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "datasets", "Drive", "augmented")
mask = 128
overlap = 25
color_black_threshold = 50
black_amount_threshold = 0.7
real_images = list()


def isMostlyBlack(pixels):
    nblack = 0
    for pixel in pixels:
        if pixel < color_black_threshold:
            nblack += 1
    n = len(pixels)
    if (nblack / float(n)) > black_amount_threshold:
        return True
    return False


def crop_imgs_from_source(dir, data_array, label):
    iterator_x = 0
    iterator_y = 0
    img_count = 0
    for filename in listdir(dir):
        if filename != 'desktop.ini':
            img_data = Image.open(dir + os.sep + filename).convert('L')
            # img_data = tifffile.imread(dir + os.sep + filename)
            x, y = img_data.size
            # x, y, _ = img_data.shape
            images_x = x // mask
            images_y = y // mask
            for iy in range(0, images_y + 1):
                for ix in range(0, images_x + 1):
                    filename_test = "img" + str(img_count)
                    cropped = img_data.crop((iterator_x, iterator_y, iterator_x + mask, iterator_y + mask))
                    # cropped = img_data[iterator_x:iterator_x + mask, iterator_y:iterator_y + mask]
                    if not isMostlyBlack(cropped.getdata()):
                        img = numpy.asarray(cropped)/255
                        data_array.append(img)
                        pyplot.imshow(img, cmap="gray")
                        pyplot.show()
                    # cropped.save(img_dir_save + os.sep + label + "_" + filename_test + ".jpg")
                    img_count = 1 + img_count
                    iterator_x = iterator_x + mask - overlap
                iterator_y = iterator_y + mask - overlap
                iterator_x = 0
            iterator_y = 0


crop_imgs_from_source(img_dir_source, real_images, "drive")
img_filename = "drive_data_" + str(mask)
print("Saving file...")
numpy.save(img_dir_save + os.sep + img_filename, real_images)
print("File saved...")