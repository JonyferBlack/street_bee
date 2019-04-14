import numpy as np
import cv2
import os
import uuid
import json
import shutil
import random
from augmentation import augment_by_rotation

debug = False


def copy_files_with_masks_in_proportion(train_img_dir, val_img_dir, train_msk_dir, val_msk_dir, proportion):
    print(train_img_dir)
    files = os.listdir(train_img_dir)
    files = random.sample(files, int(len(files) * proportion))              
    for file_name in files:
        if debug: 
            print(file_name)
        src_img = os.path.join(train_img_dir, file_name)
        src_mask = os.path.join(train_msk_dir, file_name)            
        dst_img = os.path.join(val_img_dir, file_name)
        dst_msk = os.path.join(val_msk_dir, file_name)
        os.rename(src_img, dst_img)        
        os.rename(src_mask, dst_msk)


def copy_files_in_proportion(root_dir_path, dest_dir_path, proportion):
    """
    """
    for root, dirs, files in os.walk(root_dir_path):
        for dir_ in dirs:
            dir_full_path = os.path.join(root_dir_path, dir_)
            print(dir_full_path)
            files = os.listdir(dir_full_path)
            files = random.sample(files, int(len(files) * proportion))              
            for file in files:
                if debug: 
                    print(file)
                src = os.path.join(dir_full_path, file)
                dst_full = os.path.join(dest_dir_path, dir_)
                if (not os.path.exists(dst_full)):
                    os.mkdir(dst_full)             
                dst = os.path.join(dst_full, file)
                os.rename(src, dst)            
              

def filter_outliers(root_dir_path, min_thresh, max_thresh, mean):
    """
    """
    for root, dirs, files in os.walk(root_dir_path):
        for dir_ in dirs:
            dir_full_path = os.path.join(root_dir_path, dir_)
            print(dir_full_path)
            files = os.listdir(dir_full_path)
            files_count = len(files)
            
            if files_count < min_thresh:
                shutil.rmtree(dir_full_path)
                continue
            
            continue
            
            if files_count > max_thresh:
                files = random.sample(files, int(files_count - mean))                          
            else:
                continue
                
            for file in files:
                src = os.path.join(dir_full_path, file)
                os.remove(src)


def get_cat_id(known_cats, category_name):
    """
    """
    if not category_name in known_cats:
        known_cats[category_name] = len(known_cats.keys()) 
    return known_cats[category_name]


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


def create_segmentation_masks(data_dir_path, dst_img_dir, dst_mask_dir, mean_threshold, re_size = None, augment = False, bbox_dir = None, tolerance = 10):
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
            img_downsampled_save_path = os.path.join(bbox_dir, file_name+"_downsamled.png")
            bbox_image = image.copy()
        img_new_name = file_name + '.png'
        img_save_path = os.path.join(dst_img_dir, img_new_name)
        mask_save_path = os.path.join(dst_mask_dir, img_new_name)        
        
        mask = np.zeros(image.shape, dtype=np.uint8)
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count
        
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
            #category_dir = os.path.join(dest_dir, str(get_cat_id(known_cats, category_name)))
            # if (not os.path.exists(category_dir)):
            #    os.mkdir(category_dir)
            
            #sample_filename = str(uuid.uuid4())+".png"
            #sample_path = os.path.join(category_dir, sample_filename)    
            poly = np.array(poly, dtype=np.int32)
            #cv2.fillPoly(mask, poly, ignore_mask_color)           
            
            #cv2.polylines(mask,  [poly],  False,  (0, 255, 0),  10)
            cv2.fillPoly(mask, np.int_([poly]), (255, 255, 255))
            
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
            
            #sample_filename = str(uuid.uuid4())+".png"
            #sample_path = os.path.join(category_dir, sample_filename)
            
            height, width, _ = cropped.shape
            if height < tolerance or width < tolerance:
                continue
            
            #cv2.imwrite(sample_path, cropped)
            #break
            
            if (False and augment and len(os.listdir(category_dir)) < mean_threshold):
                #augment_by_rotation(cropped, sample_filename, category_dir)
                print()
            if do_bbox:
                cv2.polylines(bbox_image,[poly], True,(148,0,211), 2)

        if not re_size == None:
            dim = re_size
            #if False and isinstance(re_size[0], numbers.Real) or isinstance(re_size[1], numbers.Real):
            #    width = int(img.shape[1] * re_size[0])
            #    height = int(img.shape[0] * re_size[1])
            #    dim = (width, height) 
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
            mask = cv2.resize(mask, dim) 
        
        cv2.imwrite(img_save_path, image)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY);
        cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(mask_save_path, mask)
        #break
        image = None
        if not do_bbox:
            continue
        #resized_image = cv2.resize(bbox_image, (300, 300)) 
        #cv2.imwrite(img_downsampled_save_path, resized_image)

    print('Kernel done')
    return known_cats

              