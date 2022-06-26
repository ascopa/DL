import os
from os import listdir

from PIL import Image, ImageChops
import numpy
from PIL import Image

# load all images in a directory
dataset = "NeuB1"
img_dir_source = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "zhao-li-cheng", "2D_Neuron_dataset", dataset, "Color_GT")
seg_dir_source = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis","DL", "zhao-li-cheng", "2D_Neuron_dataset", dataset, "Segmentation_GT")
img_dir_save = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "benchmark net", "dataset")
mask = 512
real_images = list()
seg_images = list()

def crop_imgs_from_source(dir, data_array, label):
    img_count = 0
    for filename in listdir(dir):
        if filename != 'desktop.ini':
            img_data = Image.open(dir + "\\" + filename).convert('L')
            if label == "realimg":
                img_data = ImageChops.invert(img_data)
            img_data = img_data.resize((mask, mask), Image.ANTIALIAS)
            data_array.append((numpy.array(img_data)) / 255)
            if img_count in range(10, 20):
                filename_test = "img" + str(img_count)
                img_data.save(img_dir_save + "\\" + label + "_" +filename_test + ".jpg")
            img_count += 1
            print(img_count)


crop_imgs_from_source(img_dir_source, real_images, "realimg")
crop_imgs_from_source(seg_dir_source, seg_images, "segimg")
img_filename = "real_images"
seg_filename = "segmentation_images"
print("Saving file...")
numpy.save(img_filename, real_images)
print("File saved...")
print("Saving file...")
numpy.save(seg_filename, seg_images)
print("File saved...")
