import csv
import glob
import os
import random

import cv2
import numpy as np
import scipy.stats.stats as st
from skimage.measure import label
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
#features for whole-slide image classification task
#global features
#
#   1. The ratio between the area of metastatic regions and the tissue area.
#   2. The sum of all cancer metastases probailities detected in the metastasis identification task, divided by the tissue area. caculate them at 5 different thresholds (0.5, 0.6, 0.7, 0.8, 0.9), so the total 10 global features
#
#local features
#
#Based on 2 largest metastatic candidate regions (select them based on a threshold of 0.5). 9 features were extracted from the 2 largest regions:
#
#   1. Area: the area of connected region
#   2. Eccentricity: The eccentricity of the ellipse that has the same second-moments as the region
#   3. Extend: The ratio of region area over the total bounding box area
#   4. Bounding box area
#   5. Major axis length: the length of the major axis of the ellipse that has the same normalized second central moments as the region
#   6. Max/mean/min intensity: The max/mean/minimum probability value in the region
#   7. Aspect ratio of the bounding box
#   8. Solidity: Ratio of region area over the surrounding convex area



N_FEATURES = 32
# for global features

def glob_features (slide_path, heatmap):
   
# make the heatmap path the same as slide path
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
            rgbbinaryarea = cv2.countNonZero(rgbbinary)

            predthreshold50 = heatmap > 0.5
            predthreshold60 = heatmap > 0.6
            predthreshold70 = heatmap > 0.7
            predthreshold80 = heatmap > 0.8
            predthreshold90 = heatmap > 0.9


            ratio_cancer_tissue50 =  cv2.countNonZero(predthrsold50*1)/rgbbinaryarea
            ratio_cancer_tissue60 =  cv2.countNonZero(predthrsold60*1)/rgbbinaryarea
            ratio_cancer_tissue70 =  cv2.countNonZero(predthrsold70*1)/rgbbinaryarea
            ratio_cancer_tissue80 =  cv2.countNonZero(predthrsold80*1)/rgbbinaryarea
            ratio_cancer_tissue90 =  cv2.countNonZero(predthrsold90*1)/rgbbinaryarea

            
            predthreshold250 = heatmap - 0.5
            predthreshold260 = heatmap - 0.6
            predthreshold270 = heatmap - 0.7
            predthreshold280 = heatmap - 0.8
            predthreshold290 = heatmap - 0.9
            predthreshold250 =  predthreshold250.clip(min=0)
            predthreshold260 =  predthreshold260.clip(min=0)
            predthreshold270 =  predthreshold270.clip(min=0)
            predthreshold280 =  predthreshold280.clip(min=0)
            predthreshold290 =  predthreshold290.clip(min=0)

            ratio_sum_tissue50 = predthreshold250.sum()/rgbbinaryarea
            ratio_sum_tissue60 = predthreshold260.sum()/rgbbinaryarea
            ratio_sum_tissue70 = predthreshold270.sum()/rgbbinaryarea
            ratio_sum_tissue80 = predthreshold280.sum()/rgbbinaryarea
            ratio_sum_tissue90 = predthreshold290.sum()/rgbbinaryarea

            globalfeatures = [ratio_cancer_tissue50, ratio_cancer_tissue60,ratio_cancer_tissue70,ratio_cancer_tissue80,ratio_cancer_tissue90,  ratio_sum_tissue50,  ratio_sum_tissue60,  ratio_sum_tissue70,  ratio_sum_tissue80, ratio_sum_tissue90]




    return globalfeatures


# for local features

def get_region_props(heatmapbinary, heatmap):
    heatmapbinary = closing(heatmapbinary, square[3])
    heatmapbinary = clear_border(heatmapbinary)
    labeled_img = label(heatmapbinary)
    return regionprops(labeled_img, intensity_image=heatmap)


#   1. Area: the area of connected region

def get_largest_tumor_index_area(region_props):
    largest_tumor_index = -1

    largest_tumor_area = -1

    n_regions = len(region_props)
    for index in range(n_regions):
        if region_props[index]['area'] > largest_tumor_area:
            largest_tumor_area = region_props[index]['area']
            largest_tumor_index = index

    return (largest_tumor_index, largest_tumor_area)

def get_second_largest_tumor_index_area(region_props, largest_index):
    second_largest_tumor_index = -1

    second_largest_tumor_area = -1

    n_regions = len(region_props)
    for index in range(n_regions):
        if region_props[index]['area'] > largest_tumor_area and region_props[index]['area'] < region_props[largest_index]['area']: 
            second_largest_tumor_area = region_props[index]['area']
            second_largest_tumor_index = index

    return (second_largest_tumor_index, second_largest_tumor_area)








#   Major axis length: the length of the major axis of the ellipse that has the same normalized second central moments as the region

#def get_longest_axis_in_largest_tumor_region(region_props, tumor_region_index):
#    tumor_region = region_props[tumor_region_index]
#    return max(tumor_region['major_axis_length'], tumor_region['minor_axis_length'])


def local_features(heatmap_prob):
    

    heatmapbinary = (heatmap > 0.5)*1


    localfeatures = []
    
    # extract parameters from regionprops function of scikit-image

    region_props_largest = get_region_props(heatmapbinary, heatmap)

    number_tumor_region = len(region_props_t90)

    if number_tumor_region == 0:
        return [0.00] * N_FEATURES


    #   1. Area: the area of connected region

    # the area and index of largest lesion:

    largest_lesion = get_largest_tumor_index(region_props_largest)

    largest_area = largest_lesion[1]

    largest_index = largest_lesion[0]

    features.extend(largest_area)

    #   2. Eccentricity: The eccentricity of the ellipse that has the same second-moments as the region
    eccentricity_largest = region_props_largest[largest_index]['eccentricity']

    features.extend(eccentricity_largest)

    #   3. Extend: The ratio of region area over the total bounding box area
    extent_largest = region_props_largest[largest_index]['extent']

    features.extend(extent_largest)
    
    #   4. Bounding box area
    area_bbox_largest = region_props_largest[largest_index]['bbox_area']

    features.extend(area_bbox_largest)


    #   5. Major axis length: the length of the major axis of the ellipse that has the same normalized second central moments as the region
    major_axis_length_largest = region_props_largest[largest_index]['major_axis_length']

    features.extend(major_axis_length_largest)


    #   6. Max/mean/min intensity: The max/mean/minimum probability value in the region

    maxprob_largest = region_props_largest[largest_index]['max_intensity']
    minprob_largest = region_props_largest[largest_index]['min_intensity']
    aveprob_largest = region_props_largest[largest_index]['mean_intensity']

    features.extend(maxprob_largest, minprob_largest_aveprob_largest)

    #   7. Aspect ratio of the bounding box
    coordinates_of_bbox_largest = region_props_largest[largest_index]['bbox']
    aspect_ratio_bbox_largest = (coordinates_of_bbox_largest[2]-coordinates_of_bbox_largest[0])/(coordinates_of_bbox_largest[3]-coordinates_of_bbox_largest[1])

    features.extend(aspect_ratio_bbox_largest)

    #   8. Solidity: Ratio of region area over the surrounding convex area
    solidity_largest = region_props_largest[largest_index]['solidity']

    features.extend(solidity_largest)



    #   1. Area: the area of connected region

    # the area and index of largest lesion:

    second_largest_lesion = get_second_largest_tumor_index(region_props_largest)

    second_largest_area = second_largest_lesion[1]

    second_largest_index = second_largest_lesion[0]

    features.extend(second_largest_area)

    #   2. Eccentricity: The eccentricity of the ellipse that has the same second-moments as the region
    eccentricity_second_largest = region_props_second_largest[second_largest_index]['eccentricity']

    features.extend(eccentricity_second_largest)

    #   3. Extend: The ratio of region area over the total bounding box area
    extent_second_largest = region_props_second_largest[second_largest_index]['extent']

    features.extend(extent_second_largest)
    
    #   4. Bounding box area
    area_bbox_second_largest = region_props_second_largest[second_largest_index]['bbox_area']

    features.extend(area_bbox_second_largest)


    #   5. Major axis length: the length of the major axis of the ellipse that has the same normalized second central moments as the region
    major_axis_length_second_largest = region_props_second_largest[second_largest_index]['major_axis_length']

    features.extend(major_axis_length_second_largest)


    #   6. Max/mean/min intensity: The max/mean/minimum probability value in the region

    maxprob_second_largest = region_props_second_largest[second_largest_index]['max_intensity']
    minprob_second_largest = region_props_second_largest[second_largest_index]['min_intensity']
    aveprob_second_largest = region_props_second_largest[second_largest_index]['mean_intensity']

    features.extend(maxprob_second_largest, minprob_second_largest_aveprob_second_largest)

    #   7. Aspect ratio of the bounding box
    coordinates_of_bbox_second_largest = region_props_second_largest[second_largest_index]['bbox']
    aspect_ratio_bbox_second_largest = (coordinates_of_bbox_second_largest[2]-coordinates_of_bbox_second_largest[0])/(coordinates_of_bbox_second_largest[3]-coordinates_of_bbox_second_largest[1])

    features.extend(aspect_ratio_bbox_second_largest)

    #   8. Solidity: Ratio of region area over the surrounding convex area
    solidity_second_largest = region_props_second_largest[second_largest_index]['solidity']

    features.extend(solidity_second_largest)



    return localfeatures


heatmap_path = '/home/wli/Downloads/pred/'
heatmap_paths = glob.glob(osp.join(heatmap_path, '*.npy'))
slide_path = '/home/wli/Downloads/Camelyon16/training/tumor'

cols = ['name', 'tumor', globalfeatures[0],globalfeatures[1],globalfeatures[2],globalfeatures[3],globalfeatures[4],globalfeatures[5],globalfeatures[6],globalfeatures[7],globalfeatures[8],globalfeatures[9],'largest_area','eccentricity_largest','extend_largest','area_bbox_largest','major_axis_length_largest','maxprob_largest','minprob_largest', 'aveprob_largest', 'solidity_largest','second_largest_area','eccentricity_second_largest','extend_second_largest','area_bbox_second_largest','major_axis_length_second_largest','maxprob_second_largest','minprob_second_largest', 'aveprob_second_largest', 'solidity_second_largest']

data_sheet_for_random_forest= pd.DataFrame(columns=cols)


for i in range(len(heatmap_paths)):
    heatmap = np.load(heatmap_paths[i])
    slide_path = glob.glob(osp.join(slide_path, rename(split(basename(heatmap_path[i])))))
    data_sheet_for_random_forest.at[i, 'name'] = basename(slide_path)

    slide_contains_tumor = osp.basename(slide_paths[i]).startswith('tumor_')
    if slide_contains_tumor:
        data_sheet_for_random_forest.at[i, 'tumor'] = 1

    else:

        data_sheet_for_random_forest.at[i, 'tumor'] = 0


    
    heatmapbinary_lesion = (heatmap > 0.5)*1
    number_lesion = len(get_region_props(heatmapbinary_lesion, heatmap))
    if number_lesion == 0:

        features = [0.00]*N_FEATURES

    else:

        features = glob_features(slide_path, heatmap) + local_features(heatmap)


    data_sheet_for_random_forest = data_sheet_for_random_forest.append(pd.Series(features, index=cols[2:]), ignore_index=True)








data_sheet_for_random_forest.to_csv('data_sheet_for_random_forest.csv')
