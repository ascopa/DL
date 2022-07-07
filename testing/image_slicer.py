import os
from os import listdir

import numpy
from PIL import Image

# load all images in a directory
img_dir_source_n = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "testing", "../datasets/Cancer", "N", "N", "rgb")
img_dir_source_l = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "testing", "../datasets/Cancer", "L", "L", "rgb")
img_dir_source_p = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "testing", "../datasets/Cancer", "P", "P", "rgb")
img_dir_save = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "testing", "test_img_128")
mask = 128
overlap = 10
loaded_images_arr = list()

def crop_imgs_from_source(img_dir_source, data_array, marker):
    iterator_x = 0
    iterator_y = 0
    img_count = 0
    if data_array.__len__() < 40000:
        for filename in listdir(img_dir_source):
            if filename != 'desktop.ini' and img_count < 40000:
                img_data = Image.open(img_dir_source + "\\" + filename).convert('L')
                # img_data = img_data.resize((mask, mask), Image.ANTIALIAS)
                x, y = img_data.size
                images_x = x // mask
                images_y = y // mask
                for iy in range(0, images_y + 1):
                    for ix in range(0, images_x + 1):
                        cropped = img_data.crop((iterator_x, iterator_y, iterator_x + mask, iterator_y + mask))
                        data_array.append((numpy.array(cropped)) / 255)
                        img_count = 1 + img_count
                        iterator_x = iterator_x + mask - overlap
                        # if img_count in range(50, 100):
                        #     filename_test = "img" + str(img_count)
                        #     cropped.save(img_dir_save + "\\" + marker + "_" + filename_test + ".jpg")
                        print('> loaded %s %s, crop %s' % (filename, cropped.size, img_count))
                    iterator_y = iterator_y + mask - overlap
                    iterator_x = 0
                iterator_y = 0


crop_imgs_from_source(img_dir_source_l, loaded_images_arr, "l")
crop_imgs_from_source(img_dir_source_n, loaded_images_arr, "n")
crop_imgs_from_source(img_dir_source_p, loaded_images_arr, "p")
filename_arr = "cancer_data_cropped_" + str(mask)
print("Saving file...")
numpy.save(filename_arr, loaded_images_arr)
print("File saved...")
