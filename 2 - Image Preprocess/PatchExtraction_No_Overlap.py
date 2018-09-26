#! /usr/bin/env python3
from scipy.misc import imsave
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from pathlib import Path
from skimage.filters import threshold_otsu
import glob
#before importing HDFStore, make sure 'tables' is installed by pip3 install tables
from pandas import HDFStore
from openslide.deepzoom import DeepZoomGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
#import tensorflow as tf

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)

print('Hi, patch extraction can take a while, please be patient...')
#slide_path = '/raida/wjc/CAMELYON16/training/tumor'
slide_path = '/home/wli/Downloads/CAMELYON16/training/tumor'

#anno_path = '/home/wli/Downloads/CAMELYON16/training/Lesion_annotations'
#BASE_TRUTH_DIR = '/raida/wjc/CAMELYON16/training/masking'
BASE_TRUTH_DIR = '/home/wli/Downloads/CAMELYON16/masking'

slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
slide_paths.sort()
BASE_TRUTH_DIRS = glob.glob(osp.join(BASE_TRUTH_DIR, '*.tif'))
BASE_TRUTH_DIRS.sort()
#image_pair = zip(tumor_paths, anno_tumor_paths)  
#image_pair = list(image_mask_pair)
patch_path = '/home/wli/Documents/patches'
sampletotal = pd.DataFrame([])
i=0
while i < len(slide_paths):
    #sampletotal = pd.DataFrame([])
    base_truth_dir = Path(BASE_TRUTH_DIR)
    slide_contains_tumor = osp.basename(slide_paths[i]).startswith('tumor_')
    
    with openslide.open_slide(slide_paths[i]) as slide:
        thumbnail = slide.get_thumbnail((slide.dimensions[0] / 224, slide.dimensions[1] / 224))
    
        thumbnail_grey = np.array(thumbnail.convert('L')) # convert to grayscale
        thresh = threshold_otsu(thumbnail_grey)
        binary = thumbnail_grey > thresh
    
        patches = pd.DataFrame(pd.DataFrame(binary).stack())
        patches['is_tissue'] = ~patches[0]
        patches.drop(0, axis=1, inplace=True)
        patches['slide_path'] = slide_paths[i]
    
    if slide_contains_tumor:
        truth_slide_path = base_truth_dir / osp.basename(slide_paths[i]).replace('.tif', '_mask.tif')
        with openslide.open_slide(str(truth_slide_path)) as truth:
            thumbnail_truth = truth.get_thumbnail((truth.dimensions[0] / 224, truth.dimensions[1] / 224)) 
        
        patches_y = pd.DataFrame(pd.DataFrame(np.array(thumbnail_truth.convert("L"))).stack())
        patches_y['is_tumor'] = patches_y[0] > 0
        patches_y.drop(0, axis=1, inplace=True)

        samples = pd.concat([patches, patches_y], axis=1)
        #sampletotal.append(pd.concat([patches, patches_y], axis=1))
    else:
        samples = patches
        samples['is_tumor'] = False
        #patches['is_tumor'] = False
        #sampletotal.append(patches)
            
       
        
    #if filter_non_tissue:
    samples = samples[samples.is_tissue == True] # remove patches with no tissue
    samples['tile_loc'] = list(samples.index)
    samples.reset_index(inplace=True, drop=True)
        
    sampletotal=sampletotal.append(samples, ignore_index=True)
        
    i=i+1
        
        

# randomly drop normal patches to match the number of tumor patches
idx=sampletotal.index[sampletotal['is_tumor'] == False].tolist()
drop_indices = np.random.choice(idx, 1800000, replace=False)
sampletotal_subset = sampletotal.drop(drop_indices)
# reorder the index. this is important
sampletotal_subset.reset_index(drop=True, inplace=True)

NUM_SAMPLES = 100
sampletotal_subset= sampletotal_subset.sample(NUM_SAMPLES, random_state=42)
 
sampletotal_subset.reset_index(drop=True, inplace=True)

print(sampletotal_subset.is_tumor.value_counts())

# real picture patches generation function. In the function there is save command to save numpy array da# ta as .npz format which is independent of platform. 

import cv2
from keras.utils.np_utils import to_categorical

NUM_CLASSES = 2 # not_tumor, tumor

def gen_imgs(samples, batch_size, base_truth_dir=BASE_TRUTH_DIR, shuffle=True):
   
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        if shuffle:
            samples = samples.sample(frac=1) # shuffle samples
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
        
            #images = []
            #masks = []
            for _, batch_sample in batch_samples.iterrows():
                slide_contains_tumor = osp.basename(batch_sample.slide_path).startswith('tumor_')
                 
                with openslide.open_slide(batch_sample.slide_path) as slide:
                    tiles = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=False)
                    img = tiles.get_tile(tiles.level_count-1, batch_sample.tile_loc[::-1])
                    im = np.array(img)
                    int1, int2= batch_sample.tile_loc[::-1]
                    if  batch_sample.is_tumor==True:
                        imsave('/home/wli/Documents/patches/tumor/%s_%d_%d.png' % (os.path.splitext(osp.basename(batch_sample.slide_path))[0], int1, int2), im)
                    else:
                        imsave('/home/wli/Documents/patches/normal/%s_%d_%d.png' % (os.path.splitext(osp.basename(batch_sample.slide_path))[0], int1, int2), im)

                # only load truth mask for tumor slides
                if slide_contains_tumor:
                    truth_slide_path = osp.join(base_truth_dir, osp.basename(batch_sample.slide_path).replace('.tif', '_mask.tif'))
                    with openslide.open_slide(str(truth_slide_path)) as truth:
                        truth_tiles = DeepZoomGenerator(truth, tile_size=224, overlap=0, limit_bounds=False)
                        mask = truth_tiles.get_tile(truth_tiles.level_count-1, batch_sample.tile_loc[::-1])
                        mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                        mk = np.array(mask)
                        int1, int2= batch_sample.tile_loc[::-1]
                    if  batch_sample.is_tumor==True:
                        imsave('/home/wli/Documents/patches_mask/tumor/%s_%d_%d.png' % (os.path.splitext(osp.basename(batch_sample.slide_path))[0], int1, int2), mk)
                    else:
                        imsave('/home/wli/Documents/patches_mask/normal/%s_%d_%d.png' % (os.path.splitext(osp.basename(batch_sample.slide_path))[0], int1, int2), mk) 
                else:
                    mask = np.zeros((224, 224))
                    mk = np.array(mask)
                    int1, int2= batch_sample.tile_loc[::-1]
                    imsave('/home/wli/Documents/patches_mask/normal/%s_%d_%d.png' % (os.path.splitext(osp.basename(batch_sample.slide_path))[0], int1, int2), mk)

            yield


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(sampletotal_subset, sampletotal_subset["is_tumor"]):
        train_samples_subset = sampletotal_subset.loc[train_index]
        validation_samples_subset = sampletotal_subset.loc[test_index]

BATCH_SIZE=len(train_samples_subset)

train_generator = gen_imgs(train_samples_subset, BATCH_SIZE)

next(train_generator)
