# -*- coding: utf-8 -*-
"""
Description: Move images from one folder to the other while augmenting them.
@author: kjayamanna, vmartinez, zmei
"""

#%%
import os
from shutil import copyfile
import custom_hist_final as stretch
from PIL import Image
import matplotlib.pyplot as plt

#%% create hospital directories
hospital_path = r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Spring 2022\Big Data\final_project\Datasets\Hospitals"
hospital_names = ["hospitalA", "hospitalB", "hospitalC", "hospitalD"]
img_classes = ["glioma", "meningioma", "notumor", "pituitary"]
for name in hospital_names:
    os.mkdir(os.path.join(hospital_path, name))
    for class_name in img_classes:
        os.mkdir(os.path.join(hospital_path, name, class_name))
#%%
count = 0
for hospital in hospital_names:
    for class_name in img_classes:
        src = os.path.join(
            r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Spring 2022\Big Data\final_project\Datasets\Original\equal_split",
            class_name,
        )
        dst = os.path.join(
            r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Spring 2022\Big Data\final_project\Datasets\Hospitals",
            hospital,
            class_name,
        )
        filenames = os.listdir(src)
        for idx in range(count, count + 405):
            src_file = os.path.join(src, filenames[idx])
            dst_file = os.path.join(dst, filenames[idx])
            copyfile(src_file, dst_file)
    count = count + 405
#%% create degraded images path
hospital_path = r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Spring 2022\Big Data\final_project\Datasets\degradated_hospitals"
hospital_names = ["hospitalA", "hospitalB", "hospitalC", "hospitalD"]
img_classes = ["glioma", "meningioma", "notumor", "pituitary"]
for name in hospital_names:
    os.mkdir(os.path.join(hospital_path, name))
    for class_name in img_classes:
        os.mkdir(os.path.join(hospital_path, name, class_name))

#%% Hospital A Degradations (underexposed)
img_classes = ["glioma", "meningioma", "notumor", "pituitary"]
# Input Range
a = 50
b = 100
# Output Range
c = 20
d = 60

hospital = "hospitalA"
for class_name in img_classes:
    src = os.path.join(
        r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Spring 2022\Big Data\final_project\Datasets\Hospitals",
        hospital,
        class_name,
    )
    dst = os.path.join(
        r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Spring 2022\Big Data\final_project\Datasets\degradated_hospitals",
        hospital,
        class_name,
    )
    filenames = os.listdir(src)
    for idx in range(len(filenames)):
        src_file = os.path.join(src, filenames[idx])
        # Open the file and convert to gray scale.
        img = Image.open(src_file).convert("L")
        # Generate the Histogram
        hist = stretch.histogram(img)
        # Find qp
        qp = stretch.customEq(img, hist, a, b, c, d)
        # Get the histogram plots and the equalized Image
        img_eq, fig = stretch.plot_hist(img, qp)
        im = Image.fromarray(img_eq)
        im.save(os.path.join(dst, filenames[idx]))
#%% Hospital B Degradations (overexposed)
img_classes = ["glioma", "meningioma", "notumor", "pituitary"]
# Input Range
a = 50
b = 100
# Output Range
c = 100
d = 200

hospital = "hospitalB"
for class_name in img_classes:
    src = os.path.join(
        r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Spring 2022\Big Data\final_project\Datasets\Hospitals",
        hospital,
        class_name,
    )
    dst = os.path.join(
        r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Spring 2022\Big Data\final_project\Datasets\degradated_hospitals",
        hospital,
        class_name,
    )
    filenames = os.listdir(src)
    for idx in range(len(filenames)):
        src_file = os.path.join(src, filenames[idx])
        # Open the file and convert to gray scale.
        img = Image.open(src_file).convert("L")
        # Generate the Histogram
        hist = stretch.histogram(img)
        # Find qp
        qp = stretch.customEq(img, hist, a, b, c, d)
        # Get the histogram plots and the equalized Image
        img_eq, fig = stretch.plot_hist(img, qp)
        im = Image.fromarray(img_eq)
        im.save(os.path.join(dst, filenames[idx]))
#%%
import cv2
import numpy as np


def low_pass(im):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(im, -1, kernel)


def speckle(im):
    row, col = im.shape
    gauss = np.random.randn(row, col)
    gauss = gauss.reshape(row, col)
    sigma = 0.05
    noisy = im + im * gauss * sigma
    return noisy


#%%
img_classes = ["glioma", "meningioma", "notumor", "pituitary"]
hospital = "hospitalC"
for class_name in img_classes:
    src = os.path.join(
        r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Spring 2022\Big Data\final_project\Datasets\Hospitals",
        hospital,
        class_name,
    )
    dst = os.path.join(
        r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Spring 2022\Big Data\final_project\Datasets\degradated_hospitals",
        hospital,
        class_name,
    )
    filenames = os.listdir(src)
    for idx in range(len(filenames)):
        src_file = os.path.join(src, filenames[idx])
        # Open the file and convert to gray scale.
        img = cv2.imread(src_file, 0)
        degraded = low_pass(img)
        degraded = speckle(degraded)
        degraded = Image.fromarray(degraded)
        degraded.convert("L").save(os.path.join(dst, filenames[idx]))
