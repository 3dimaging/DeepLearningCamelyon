# Questions
1. Where do I store papers for references?
2. For augmentations - I'm currently running:
            Image data generator -> Random combo of:
            Rotate =< 20-degrees
            vert & horiz flip TRUE
            shearing =< 20% TRUE
   I'm not understanding how to change augmentations. I think you mean different parts of aug so table looks something like below- yes?:
   
   |Test|Aug v.1|Aug v.2|Aug v.3|Aug v.4|Aug v.5|Aug v.6|
   |--|--|--|--|--|--|--|
   |i|||||||
   |ii|||||||
   |...|||||||
   |viii||||||||
   
   Latest instructions: "Great, so since augmentations helped, I think we should make a table with the results as a number of the number of augmentations.  Let’s keep the same differences between the 1+, 2+, 3+ but explore more this effect of the augmentations.  Maybe a couple more angles of rotation, different flip?"
3. Need to insert data augmentation into the k-fold.... figure this out. 'flow from directory' will NOT work here... will need to find a new function. no 'save to' command, unless I need to save it to a temporary directory....
           - if I quintouple all '0's in a training set then I would most often, have 305 data points. I would still only have output on 305 data points, but I would actually train sometimes on only 300 data points. Is this bad???
           - which is better: Holdout set and all data augmented in basepath, OR LOO and data augmented within each fold???

# To Do
1. HER2 Testing Progress:
   10. Pulling from three different scanners for test/train set -> need to figure this out then set it up for tests x-xiii
       Remaining testing scenarios listed in multi-scanner 2-fold test scenario folder.

2. Documentation:
   1. Begin writing notes in the format of a scientific paper, as discussed with Marios.
   2. Finish transferring notes from notebook to written document
   3. Combine existing note documents to new format
   
3. GitHub:
   1. Currently feeling like folder naming looks messy- need to re-evaluate
   2. Switch to pycharm to more easily add to github? 
   3. Walk through github tutorials to start hosting on local and autouploading to git
   
4. References:
   1. Ask Marios about where to store papers for references. 
   2. Investigate image augmentation experiments further
   
5. Jerry:
   1. re-review Cluster and ReadMe documentation
   2. Make edits to cluster and readme vignettes
   3. Review changes to local doc -> ensure everything looks good to send to Brandon for first pass
   4. Find "check components working" document where all functions in package are checked and confirmed running correctly
   
6. Maintenance:
   1. Talk with Jonathan to figure out my RaidA situation -> start adding things to Marios' folder
   2. Make sure Marios can get on Git. Add a new repository for him??? Talk to Brandon about making/etc.




# Notes
 - Out of Office Thursday & Friday, 9/20-9/21
