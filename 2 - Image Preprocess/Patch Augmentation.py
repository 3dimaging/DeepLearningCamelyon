import os
from scipy.ndimage import filters
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from pathlib import Path
from scipy.misc import imsave
from skimage.filters import threshold_otsu
import glob
#before importing HDFStore, make sure 'tables' is installed by pip3 install tables
from pandas import HDFStore
#from openslide.deepzoom import DeepZoomGenerator
from skimage.filters import threshold_otsu
import cv2 as cv2
import staintools
from skimage import io
import math

#for 224 images
tumor_paths = glob.glob(osp.join('/home/wli/Downloads/test/tumor/', '*.png'))
tumor_paths.sort()
mask_paths = '/home/wli/Downloads/test/mask/'
#mask_paths.sort()

def noise_generator (noise_type,image):
    """
    Generate noise to a given Image based on required noise type
    
    Input parameters:
        image: ndarray (input image data. It will be converted to float)
        
        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row,col,ch= image.shape
    if noise_type == "gauss":       
        mean = 0.0
        var = 0.01
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.0005
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    else:
        return image

    
def gaussnoise(image):  
    row,col,ch= image.shape
    mean = 0.0
    var = 0.01
    sigma = var**0.5
    gauss = np.array(image.shape)
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy.astype('uint8')

def random_crop2(image, mask, crop_size):
    
    #width, height = slide.level_dimensions[4]
    dy, dx = crop_size
    x = np.random.randint(0, 256 - dx + 1)
    y = np.random.randint(0, 256 - dy + 1)
    index=[x, y]
    #cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    cropped_img = image[x:(x+dx), y:(y+dy),:]
    #cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    cropped_mask = mask[x:(x+dx), y:(y+dy)]
    
    return (cropped_img, cropped_mask, index)

crop_size2=[224, 224]        
i=0
while i < len(tumor_paths):
        tumor_path = tumor_paths[i] 
        mask_path = osp.join(mask_paths, osp.basename(tumor_paths[i].replace('.png', '_mask.png')))
        #image = plt.imread(tumor_path)
        imgmask = io.imread(mask_path)

        stain_normalizer = staintools.StainNormalizer(method='vahadane')
        imagest = staintools.read_image("/home/wli/Downloads/test/tumor_st.png")
        img = staintools.read_image(tumor_path)
        standardizer = staintools.BrightnessStandardizer()
        imagest_standard = standardizer.transform(imagest)
        img_standard = standardizer.transform(img)
        stain_normalizer.fit(imagest_standard)
        img_norm = stain_normalizer.transform(img_standard)


        imageroted1=i1_flip=np.fliplr(img_norm)
        maskroted1=i1_flip=np.fliplr(imgmask)
          
        imageroted2=np.rot90(img_norm, 1)
        imageroted3=np.rot90(img_norm, 2)
        imageroted4=np.rot90(img_norm, 3)
          
        maskroted2=np.rot90(imgmask, 1)
        maskroted3=np.rot90(imgmask, 2)
        maskroted4=np.rot90(imgmask, 3)
          
        maskset  = [imgmask, maskroted1, maskroted2, maskroted3, maskroted4]
        imageset  = [img_norm, imageroted1,imageroted2, imageroted3, imageroted4]
        
        for m in range(len(imageset)):
            crop224 = random_crop2(imageset[m],maskset[m], crop_size2)
            imsave('/home/wli/Downloads/test/augtumor/%s_%d_%d.png' % (osp.splitext(osp.basename(tumor_path))[0], crop224[2][0], crop224[2][1]), crop224[0])
            io.imsave('/home/wli/Downloads/test/augmask/%s_%d_%d_mask.png' % (osp.splitext(osp.basename(mask_path))[0], crop224[2][0], crop224[2][1]), crop224[1])
            crop224 = random_crop2(imageset[m],maskset[m], crop_size2)
            imsave('/home/wli/Downloads/test/augtumor/%s_%d_%d.png' % (osp.splitext(osp.basename(tumor_path))[0], crop224[2][0], crop224[2][1]), crop224[0])
            io.imsave('/home/wli/Downloads/test/augmask/%s_%d_%d_mask.png' % (osp.splitext(osp.basename(mask_path))[0], crop224[2][0], crop224[2][1]), crop224[1])
   
        
       
          
        imgnoise = []
        for img in imageset:
            imgnoise.append(noise_generator('s&p', img))
            
        
        
        
        
            
        for s in range(len(imgnoise)):
            crop224n = random_crop2(imgnoise[s],maskset[s], crop_size2)
            imsave('/home/wli/Downloads/test/augtumor/%s_%d_%d_n.png' % (osp.splitext(osp.basename(tumor_path))[0], crop224n[2][0], crop224n[2][1]), crop224n[0])
            io.imsave('/home/wli/Downloads/test/augmask/%s_%d_%d__n_mask.png' % (osp.splitext(osp.basename(mask_path))[0], crop224n[2][0], crop224n[2][1]), crop224n[1])
            crop224n = random_crop2(imgnoise[s],maskset[s], crop_size2)
            imsave('/home/wli/Downloads/test/augtumor/%s_%d_%d_n.png' % (osp.splitext(osp.basename(tumor_path))[0], crop224n[2][0], crop224n[2][1]), crop224n[0])
            io.imsave('/home/wli/Downloads/test/augmask/%s_%d_%d_n_mask.png' % (osp.splitext(osp.basename(mask_path))[0], crop224n[2][0], crop224n[2][1]), crop224n[1])

        


        i=i+1
