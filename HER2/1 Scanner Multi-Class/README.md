# This is the First pass at the HER2 Multi-class single scanner test

Each test runs 5-fold (no holdout) for data from only one scanner (single subset) 

Next: 
1. Print scores for every image. 
2. Move into multi-scanner.












Need to evaluate:
1. Is there a way to predict accuracy with greater than 2 significant figures?
2. "pred" object used to produce [0,1,2], now produce [1,2] - why??
3. Should we run this with a holdout set? If yes, is the holdout set different for each subset(scanner)?
4. Should I save model weights? Is there any reason I will need to use an identical system later?


