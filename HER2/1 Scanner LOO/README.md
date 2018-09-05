# Leave One Out method applied to train/test split

**Problem:** Not predicting '0' class in k-fold=2

**Solution:** Ensure at least 20 of 21 total '0' class images are in train set

## Results:

||0|1|2|Tot|
| :--: | :--: | :--: | :--: | :--: |
|**0**|2|19|-|21|
|**1**|-|125|12|137|
|**2**|-|13|70|83|
|**Tot**|2|157|82|241|

% Agreement: 81%
  - '0' = 10%
  - '1' = 91%
  - '2' = 84%




'0':  `PPV:` 1.0 `NPV:` .92
   
'1': `PPV:` .80 `NPV:` .90

'2': `PPV:` .85 `NPV:` .92

