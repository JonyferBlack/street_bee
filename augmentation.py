import numpy as np
import cv2
import imutils
import os

debug = False


def augment_by_rotation(img, file_path, dst_dir, ops_count=6, ang_min=-23, ang_max=23, crop_coeff = 1, to_small = 15):
    """
    """
    step = int((ang_max - ang_min) / ops_count)
    for angle in np.arange(ang_min, ang_max, step):
        src_height, src_width, _ = img.shape
        src_width *= crop_coeff
        src_height *= crop_coeff
        dst = imutils.rotate_bound(img, angle)
        dst_height, dst_width, _ = dst.shape
        width = int(min(dst_width, src_width))
        height = int(min(dst_height, src_height))
        x = int((max(dst_width, src_width) - width) // 2)
        y = int((max(dst_height, src_height) - height)// 2)
        if (width - x) > to_small and (height - y) > to_small:
            dst = dst[y : height, x : width].copy()
        aug_sample_filename = os.path.join(dst_dir, 'ang_'+str(angle)+'_'+file_path)
        cv2.imwrite(aug_sample_filename, dst)


def augment_by_noise(file_path, src_dir, dst_dir, ops_count=3, str_min=15, str_max=35):
    """
    """
    step = int((str_max - str_min) / ops_count)
    for strength in np.arange(str_min, str_max, step):
        strength = int(strength)
        #print(strength)
        im_path = os.path.join(src_dir, file_path)
        #print(im_path)
        sample = cv2.imread(im_path)
        rand = sample.copy()
        m = (strength, strength, strength) 
        s = (strength, strength, strength)
        cv2.randn(rand, m, s)
        dst = cv2.bitwise_or(sample, rand)
        aug_sample_filename = os.path.join(dst_dir, str(strength)+'_'+file_path)
        cv2.imwrite(aug_sample_filename, dst)    


def augment_dataset_by_noise(train_dir, threshold, min_thresh):
    """
    """
    for root, dirs, files in os.walk(train_dir):
        for dir_ in dirs:
            print(dir_)
            dir_full_path = os.path.join(train_dir, dir_)
        
            files = os.listdir(dir_full_path)
    
            for file in files:
                if debug: 
                    print(file)
                augment_by_noise(file, dir_full_path, dir_full_path)
                

def augment_dataset_by_angle(train_dir):
    """
    """
    for root, dirs, files in os.walk(train_dir):
        for dir_ in dirs:
            print(dir_)
            dir_full_path = os.path.join(train_dir, dir_)
            
            files = os.listdir(dir_full_path)
    
            for file in files:
                if debug: 
                    print(file)
                img = cv2.imread(os.path.join(dir_full_path, file))
                augment_by_rotation(img, file, dir_full_path)
                

