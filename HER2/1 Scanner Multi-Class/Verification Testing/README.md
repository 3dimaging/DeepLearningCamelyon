# Verification Testing of Single Scanner Multi-Class Deep Learning System

Testing listed as follows:

## 1. Testing Change in Accuracy given Incorrectly labeled data
1. Aperio_FDA -> 10 images label 1 -> label 3
    - **Results:** ~0.8% increase in accuracy per fold
2. Aperio_NIH -> 10 images label 2 -> label 1
    - **Results:** ~5% drop in accuracy per fold
3. Hamamatsu_2 -> 10 images label 3 -> label 1
    - **Results:** ~10% drop in accuracy per fold
    
**Conclusion:** System running properly

Table 1: Accuracy per fold with Truth and altered Truth Validation Testing


|Scanner|True|Epoch|Val_test|Epoch| \( \Delta \) Acc.|Avg. \(Delta) Acc.|Description|
|-------|----|-----|--------|-----|-----------|----------------| :-------: |
|FDA|0.7551|112|0.7755|63|+0.0204|+0.00824|10 images label 1 -> label 3|
| |0.7083|93|0.75|72|+0.0417|~ +0.08%||
| |0.7279|69|0.8125|62|+0.0833|||
| |0.8542|53|0.8125|53|-0.0417|||
| |0.8333|65|0.7708|66|-0.0625|||
|NIH|0.7755|107|0.6939|81|-0.0816|-0.04966|10 images label 2 -> label 1|
| |0.8333|81|0.8125|60|-0.0208|~ -5%||
| |0.8125|109|0.8333|74|+0.0208|||
| |0.8542|74|0.7083|33|-0.1459|||
| |0.8333|62|0.8125|71|-0.0208|||
|Hama|0.8367|132|0.7347|76|-0.102|-0.09958|10 images label 3 -> label 1|
| |0.8542|93|0.8125|90|-0.0417|~ -10%||
| |0.75|61|0.7292|75|-0.0208|||
| |0.875|63|0.5833|85|-0.2917|||
| |0.8125|75|0.7708|52|-0.0417|||



LaTex for delta symbol not working :-1: 
