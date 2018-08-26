# Robustness of Deep Learning System assesing HER2 Classification
Authors: Sarah Dudgeon, Marios Gavrielides

## Objecives 
- Develop/implement DL system for HER2 classification
- Train and test on datasets from different whole slide imaging scanners to assess robustness of DL
- Examine performance of DL for classifying HER2 expression in terms of 
    a) agreement with a pathologist panel
    b) performance comparison to previously developed algorithms
    
## Methods
# 1. Database collection/description
- 64 breast cancer tissue slides stained with HER2 were scanned with 3 different WSI scanners
      - Aperio Scanscope T2 - FDA -> Subset 1
      - Aperio Scanscope T2 - NIH -> Subset 2
      - Hamamatsu NanoZoomer 2.0 -> Subset 3
      
- The same 241 regions of interest (800x600) were extracted from each subset
- Ground truth created from panel of 7 pathologists, each individually scoring every ROI from one scanner on continuous scale (0-100)


# 2. Deep Learning system
- LeNet Model (sequential feedforward Convolutional Neural Network)
    - Keras and other packages with Tensor Flow backend
    - 10 layers (3 Convolution; 3 MaxPooling; 2 Dense; 1 Flatten; 1 Dropout)
    - Patch-based trained
    - Current model subject to change
    

# 3. Experiments and Analysis
- Experiments
    - Train & Test on one Subset (3x)
    - Train (Subset 1) & Test (Subset 2-3) (3x)
    - Train across different DL variables (augmentation, parameters (model type and size), hyperparameters (dropout, epochs))
    - Examine effect of data type (continuous vs. categorical)
    - Examine effect of data truth definition (mean scores, multi-label truthing)
    
    
- Analysis
    - Compare performance among experiments (robustness of DL system given varying 
    - Understand effect & rank impact of DL parameters


