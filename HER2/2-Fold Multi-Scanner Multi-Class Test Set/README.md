# Testing Outline
1. No augmentations:
   1. Training/testing 50/50 split, all from Scanner 1 > Output=accuracy compared to classified mean score
   2. Training/testing 50/50 split, all from Scanner 2 > Output=accuracy compared to classified mean score
   3. Training/testing 50/50 split, all from Scanner 3 > Output=accuracy compared to classified mean score
   4. Training/testing 50/50 split, training from1, testing from 2 > Output=accuracy compared to classified mean score
   5. Training/testing 50/50 split, training from1, testing from 3 > Output=accuracy compared to classified mean score
   6. Training/testing 50/50 split, training from2, testing from 1 > Output=accuracy compared to classified mean score
   7. Training/testing 50/50 split, training from2, testing from 3 > Output=accuracy compared to classified mean score
   8. Training/testing 50/50 split, training from3, testing from 1 > Output=accuracy compared to classified mean score
   9. Training/testing 50/50 split, training from3, testing from 2 > Output=accuracy compared to classified mean score
   10. Training/testing 50/50 split, training from all 3 scanners, testing from 1 > Output=accuracy compared to mean score
   11. Training/testing 50/50 split, training from all 3 scanners, testing from 2 > Output=accuracy compared to mean score
   12. Training/testing 50/50 split, training from all 3 scanners, testing from 3 > Output=accuracy compared to mean score
   13. Training/testing 50/50 split, training from all 3 scanners, testing from all 3 > Output=accuracy compared to mean score
       - After all are running properly, we will do repeat (with re-shuffling/alternate seed) runs
2. Basic augmentations: (horizontal & vertical flips, basic rotation)
   1. Repeat i-xiii above
   
3. Targeted augmentations: (focusing on color properties)
   1. Repeat steps 1-2

### Dataset Details
|Subset|Source|
| :--: | :--: |
|Scanner 1|FDA|
|Scanner 2|NIH|
|Scanner 3|Hamamatsu|

 - 2-Fold x-val
   - randomization = TRUE 
   - random seed = 5
