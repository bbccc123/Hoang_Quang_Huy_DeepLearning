import os
import cv2
import numpy as np
import random

def horizontal_flip(image):
    return cv2.flip(image, 1)

def random_crop(image):
    h, w = image.shape[:2]
    zoom_factor = random.uniform(0.4, 1.0)
    cropped_h, cropped_w = int(h * zoom_factor), int(w * zoom_factor)
    y1 = random.randint(0, h - cropped_h)
    x1 = random.randint(0, w - cropped_w)
    return image[y1:y1+cropped_h, x1:x1+cropped_w]

def random_rotation(image):
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h))

def random_blur(image):
    blur_amount = random.uniform(0, 1.5)
    return cv2.GaussianBlur(image, (5, 5), blur_amount)

def preprocess_image(image):
    resized_image = cv2.resize(image, (320, 320))
    return resized_image

def augment_and_preprocess(image):
    augmented_image = image.copy()
    
    if random.random() < 0.5:
        augmented_image = horizontal_flip(augmented_image)
    augmented_image = random_crop(augmented_image)
    augmented_image = random_rotation(augmented_image)
    augmented_image = random_blur(augmented_image)
    
    preprocessed_image = preprocess_image(augmented_image)
    
    return preprocessed_image

def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"): # Only process jpg or png files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            img = cv2.imread(input_path)
            
            output_img = augment_and_preprocess(img)
            
            cv2.imwrite(output_path, output_img)

input_folder = "C:/Users/duyho/Downloads/yolov8/yolov8/train/images"
output_folder = "C:/Users/duyho/Desktop/320x320"

process_images_in_folder(input_folder, output_folder)
