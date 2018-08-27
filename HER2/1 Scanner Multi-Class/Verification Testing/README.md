# Verification Testing of Single Scanner Multi-Class Deep Learning System

Testing listed as follows:

## 1. Testing Change in Accuracy given Incorrectly labeled data
1. Aperio_FDA -> 10 images label 1 -> label 3
    - **Results:** ~0.8% increase in accuracy per fold
2. Aperio_NIH -> 10 images label 2 -> label 1
    - **Results:** ~5% drop in accuracy per fold
3. Hamamatsu_2 -> 10 images label 3 -> label 1
    - **Results:** ~10% drop in accuracy per fold

Table 1: Accuracy per fold with Truth and altered Truth Validation Testing


 |True|Epoch||Val_test|Epoch
FDA|0.7551|112||0.7755|63
 |0.7083|93||0.75|72




