import numpy as np
import cv2
import os
import uuid
import imutils
import json


debug = False


def get_cat_id(known_cats, category_name):
    if not category_name in known_cats:
        known_cats[category_name] = len(known_cats.keys()) 
    return known_cats[category_name]


def augment_by_rotation(img, file_path, dst_dir, ops_count=6, ang_min=-23, ang_max=23, crop_coeff = 1, to_small = 15):
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
                


def crop_by_markup(data_dir_path, dest_dir, mean_threshold, augment = False, bbox_dir=None, tolerance = 10):
    ""
    ""
    poly_key = 'polygon'
    rect_key = 'rect'
    category_key= 'sku_id'
    
    do_bbox = bbox_dir != None
    known_cats= {}
    json_paths = [filename for filename in os.listdir(data_dir_path) if filename.endswith(".json")]
    
    for json_file in json_paths:
        json_path = os.path.join(data_dir_path, json_file)
        with open(json_path, 'r') as f:
                markup_dict = json.load(f)
        file_name = os.path.splitext(json_file)[0]
    
        print(file_name)

        image_name = file_name+".jpg"
        image_path = os.path.join(data_dir_path, image_name)
        image = cv2.imread(image_path)
                
        if do_bbox:
            img_save_path = os.path.join(bbox_dir, image_name)
            img_downsampled_save_path = os.path.join(bbox_dir, file_name+"_downsamled.jpg")
            bbox_image = image.copy()
                
        for object_ in markup_dict['objects']:
            if (not category_key in object_):
                print("No category: ")
                print(object_)
                continue

            poly = None 
        
            if (poly_key in object_):
                poly = np.array(object_[poly_key], np.int32)
            if (rect_key in object_):
                rect = np.array(object_[rect_key], np.int32)
                poly = np.array([[rect[0][0],rect[0][1]],[rect[1][0],rect[0][1]],[rect[1][0],rect[1][1]],[rect[0][0],rect[1][1]]]) 
    
            poly = np.flip(poly, 1)
       
            category_name = object_[category_key]
            category_dir = os.path.join(dest_dir, str(get_cat_id(known_cats, category_name)))
            if (not os.path.exists(category_dir)):
                os.mkdir(category_dir)
              
            rect = cv2.boundingRect(poly)
            x,y,w,h = rect
            cropped = image[y:y+h, x:x+w].copy()

            if cropped.size==0:
                print("Image was empty:")
                print("poly:")
                print(poly)
                print('object:')
                print(object_)
                continue
            
            sample_filename = str(uuid.uuid4())+".jpg"
            sample_path = os.path.join(category_dir, sample_filename)
            
            height, width, _ = cropped.shape
            if height < tolerance or width < tolerance:
                continue
            
            cv2.imwrite(sample_path, cropped)
            
            if (augment and len(os.listdir(category_dir)) < mean_threshold):
                augment_by_rotation(cropped, sample_filename, category_dir)
            
            if do_bbox:
                cv2.polylines(bbox_image,[poly], True,(148,0,211), 2)
    
        image = None
        if not do_bbox:
            continue
    
        cv2.imwrite(img_save_path, bbox_image)
        resized_image = cv2.resize(bbox_image, (300, 300)) 
        cv2.imwrite(img_downsampled_save_path, resized_image)

    print('Kernel done')
    return known_cats
