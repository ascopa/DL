import shutil
from os import listdir

import cv2
import os
import albumentations as A
import numpy
from PIL import Image
from matplotlib import pyplot

img_dir_source = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "datasets", "Drive", "sources")
img_dir_save = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "datasets", "Drive", "augmented")

mask = 256
overlap = 30
color_black_threshold = 50
black_amount_threshold = 0.3
real_images = list()

transform1 = A.Compose([
    A.HorizontalFlip(p=1)
])

transform2 = A.Compose([
    A.RandomRotate90(p=1)
])

transform3 = A.Compose([
    A.HorizontalFlip(p=1),
    A.RandomRotate90(p=1)
])


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
            x, y = img_data.size
            images_x = x // mask
            images_y = y // mask
            for iy in range(0, images_y + 1):
                for ix in range(0, images_x + 1):
                    filename_test = "img" + str(img_count)
                    cropped = img_data.crop((iterator_x, iterator_y, iterator_x + mask, iterator_y + mask))
                    # cropped = img_data[iterator_x:iterator_x + mask, iterator_y:iterator_y + mask]
                    if not isMostlyBlack(cropped.getdata()):
                        img = numpy.asarray(cropped) / 255
                        data_array.append(img)
                        # pyplot.imshow(img, cmap="gray")
                        # pyplot.show()
                        # cropped.save(img_dir_save + os.sep + label + "_" + filename_test + ".jpg")
                    img_count = 1 + img_count
                    iterator_x = iterator_x + mask - overlap
                iterator_y = iterator_y + mask - overlap
                iterator_x = 0
            iterator_y = 0


def apply_transforms(image, data_array):
    img1 = transform1(image=image)["image"]
    img2 = transform2(image=image)["image"]
    img3 = transform3(image=image)["image"]
    data_array.append(img1)
    data_array.append(img2)
    data_array.append(img3)
    # pyplot.imshow(img1, cmap="gray")
    # pyplot.show()
    # pyplot.imshow(img2, cmap="gray")
    # pyplot.show()
    # pyplot.imshow(img3, cmap="gray")
    # pyplot.show()


def augment_images(data_array):
    aux_list = list()
    img_count = 0
    for image in data_array:
        apply_transforms(image, aux_list)
        img_count = img_count + 1
        print("Transforming image " + str(img_count))
    return [*data_array, *aux_list]


shutil.rmtree(img_dir_save)
os.mkdir(img_dir_save, 0o666)

crop_imgs_from_source(img_dir_source, real_images, "drive")
augmented_images = augment_images(real_images)
img_filename = "drive_data_" + str(mask)
print("Saving file...")
numpy.save(img_dir_save + os.sep + img_filename, augmented_images)
print("Amount of images: " + str(augmented_images.__len__()))
print("File saved...")
