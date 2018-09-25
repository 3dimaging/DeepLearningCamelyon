from matplotlib import cm
from tqdm import tqdm
from skimage.filters import threshold_otsu
from keras.models import load_model
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
import cv2
from keras.utils.np_utils import to_categorical

output_dir = Path('/home/wli/Downloads/camelyontestonly')

import os.path as osp
import openslide
from pathlib import Path
from keras.models import Sequential
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

#BASE_TRUTH_DIR = Path('/home/wli/Downloads/camelyontest/mask')

slide_path = '/home/wli/Downloads/googlepred/'
#truth_path = str(BASE_TRUTH_DIR / 'tumor_026_Mask.tif')
#slide_paths = list(slide_path)
slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
#slide = openslide.open_slide(slide_path)


def find_patches_from_slide(slide_path, filter_non_tissue=True):
    """Returns a dataframe of all patches in slide
    input: slide_path: path to WSI file
    output: samples: dataframe with the following columns:
        slide_path: path of slide
        is_tissue: sample contains tissue
        is_tumor: truth status of sample
        tile_loc: coordinates of samples in slide
        
    
    option: base_truth_dir: directory of truth slides
    option: filter_non_tissue: Remove samples no tissue detected
    """

        #sampletotal = pd.DataFrame([])
        #base_truth_dir = Path(BASE_TRUTH_DIR)
        #anno_path = Path(anno_path)
        #slide_contains_tumor = osp.basename(slide_paths[i]).startswith('tumor_')
    print (slide_path)

    dimensions = []
    
    with openslide.open_slide(slide_path) as slide:
            dtotal = (slide.dimensions[0] / 224, slide.dimensions[1] / 224)
            thumbnail = slide.get_thumbnail((dtotal[0], dtotal[1]))
            thum = np.array(thumbnail)
            ddtotal = thum.shape
            dimensions.extend(ddtotal)
            hsv_image = cv2.cvtColor(thum, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_image)
            hthresh = threshold_otsu(h)
            sthresh = threshold_otsu(s)
            vthresh = threshold_otsu(v)
        # be min value for v can be changed later
            minhsv = np.array([hthresh, sthresh, 70], np.uint8)
            maxhsv = np.array([180, 255, vthresh], np.uint8)
            thresh = [minhsv, maxhsv]
        #extraction the countor for tissue

            rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])
            _, contours, _ = cv2.findContours(rgbbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxtcols = ['xmin', 'xmax', 'ymin', 'ymax']
            bboxt = pd.DataFrame(columns=bboxtcols)
            for c in contours:
                (x, y, w, h) = cv2.boundingRect(c)
                bboxt = bboxt.append(pd.Series([x, x+w, y, y+h], index = bboxtcols), ignore_index=True)
                bboxt = pd.DataFrame(bboxt)
            
            xxmin = list(bboxt['xmin'].get_values())
            xxmax = list(bboxt['xmax'].get_values())
            yymin = list(bboxt['ymin'].get_values())
            yymax = list(bboxt['ymax'].get_values())

            xxxmin = np.min(xxmin)
            xxxmax = np.max(xxmax)
            yyymin = np.min(yymin)
            yyymax = np.max(yymax)

            dcoord = (xxxmin, xxxmax, yyymin, yyymax)

            dimensions.extend(dcoord)

           # bboxt = math.floor(np.min(xxmin)*256), math.floor(np.max(xxmax)*256), math.floor(np.min(yymin)*256), math.floor(np.max(yymax)*256)
           
            samplesnew = pd.DataFrame(pd.DataFrame(np.array(thumbnail.convert('L'))))
            print(samplesnew)
            # very critical: y value is for row, x is for column
            samplesforpred = samplesnew.loc[yyymin:yyymax, xxxmin:xxxmax]

            dsample = samplesforpred.shape

            dimensions.extend(dsample)

            np.save ('dimensions_%s' % (osp.splitext(osp.basename(slide_paths[i]))[0]), dimensions)

            print(samplesforpred)

            samplesforpredfinal = pd.DataFrame(samplesforpred.stack())

            print(samplesforpredfinal)

            samplesforpredfinal['tile_loc'] = list(samplesforpredfinal.index)

            samplesforpredfinal.reset_index(inplace=True, drop=True)


            samplesforpredfinal['slide_path'] = slide_paths[i]


            print(samplesforpredfinal)


    return samplesforpredfinal




           
    
       

import cv2
from keras.utils.np_utils import to_categorical

NUM_CLASSES = 2 # not_tumor, tumor

def gen_imgs(samples, batch_size, shuffle=False):
    """This function returns a generator that 
    yields tuples of (
        X: tensor, float - [batch_size, 224, 224, 3]
        y: tensor, int32 - [batch_size, 224, 224, NUM_CLASSES]
    )
    
    
    input: samples: samples dataframe
    input: batch_size: The number of images to return for each pull
    output: yield (X_train, y_train): generator of X, y tensors
    
    option: base_truth_dir: path, directory of truth slides
    option: shuffle: bool, if True shuffle samples
    """
    
    num_samples = len(samples)
    print(num_samples)
    
        
    images = []
        
    for _, batch_sample in batch_samples.iterrows():
                
        with openslide.open_slide(batch_sample.slide_path) as slide:
            tiles = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=False)
            print(batch_sample.tile_loc[::], batch_sample.tile_loc[::-1])
            img = tiles.get_tile(tiles.level_count-1, batch_sample.tile_loc[::-1])
                    
                
        images.append(np.array(img))

    X_train = np.array(images)
            
    yield X_train




def predict_batch_from_model(patches, model):
    
    predictions = model.predict(patches)
    #print(predictions[:, 1])
    #print(predictions[:, 0])
    predictions = predictions[:, 1]
    return predictions




model = load_model('/home/wli/Downloads/googlenet0917-02-0.93.hdf5')
alpha = 0.5

#slide = openslide.open_slide(slide_paths[0])

#n_cols = int(slide.dimensions[0] / 224)
#n_rows = int(slide.dimensions[1] / 224)
#assert n_cols * n_rows == n_samples

#thumbnail = slide.get_thumbnail((n_cols, n_rows))
#thumbnail = np.array(thumbnail)

# batch_size = n_cols
batch_size = 32


crop_size = [224, 224]
i=0
while i < len(slide_paths):
    
    output_thumbnail_preds = list()
    all_samples = find_patches_from_slide(slide_paths[i], filter_non_tissue=False)
    n_samples = len(all_samples)
    
    for offset in tqdm(list(range(0, n_samples, batch_size))):
        batch_samples = all_samples.iloc[offset:offset+batch_size]
        #png_fnames = batch_samples.tile_loc.apply(lambda coord: str(output_dir / ('%d_%d.png' % coord[::-1])))
    
        X = next(gen_imgs(batch_samples, batch_size, shuffle=False))
    
    
        preds = predict_batch_from_model(X, model)

        output_thumbnail_preds.extend(preds)
        print(output_thumbnail_preds)
        
        # overlay preds
        # save blended imgs
        #for i, png_fname in enumerate(png_fnames):
        #    pred_i = preds[i]
        #    X_i = X[i]
        #    output_img = cv2.cvtColor(X_i, cv2.COLOR_RGB2GRAY)
        #    output_img2 = cv2.cvtColor(output_img.copy(), cv2.COLOR_GRAY2RGB)

        #    overlay = np.uint8(cm.jet(pred_i) * 255)[:,:,:3]
        #    blended = cv2.addWeighted(overlay, alpha, output_img2, 1-alpha, 0, output_img)
            
            #plt.imsave(png_fname, blended)
        

    #output_thumbnail_preds = np.array(output_thumbnail_preds)

    np.save('outputheatmap_%s' % (osp.splitext(osp.basename(slide_paths[i]))[0]), output_thumbnail_preds)

    i = i+1 
