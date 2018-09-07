# Leave One Out method applied to train/test split

**Problem:** Not predicting '0' class in k-fold=2

**Solution:** 1. Ensure at least 20 of 21 total '0' class images are in train set
              2. Augment '0' label data to bolster quantity to match other labels (fix data inequality problem)
## Results:
|*LOO OG Data*|||||*LOO Augmented '0' label*|||||
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
||**0**|**1**|**2**|**Tot**||**0**|**1**|**2**|**Tot**|
|**0**|2|19|-|21|**0**|64|21|-|85|
|**1**|-|125|12|137|**1**|4|126|7|137|
|**2**|-|13|70|83|**2**|-|14|69|83|
|**Tot**|2|157|82|241|**Tot**|68|161|78|305|

LOO OG Data:
% Agreement: 81%                                
  - '0' = 10%                                   
  - '1' = 91%                                   
  - '2' = 84%                                   




'0':  `PPV:` 1.0 `NPV:` .92
   
'1': `PPV:` .80 `NPV:` .90

'2': `PPV:` .85 `NPV:` .92


LOO Augmented '0' Label:
% Agreement: 84%
  - '0' = 75%
  - '1' = 92%
  - '2' = 83%




'0': `PPV:` .94 `NPV:` .91
   
'1': `PPV:` .78 `NPV:` .94

'2': `PPV:` .88 `NPV:` .94





