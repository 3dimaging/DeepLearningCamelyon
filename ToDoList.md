# 08-28-18 patch augmentation
           How to do add color noise?
           camelyon 17 dataset will be all uploaded tomorrow
# 08-29-18 color noise
           I think I figured out color noise. 
           Also, the sequence of augmentation is important. For example, add color noise first and do flip second. The pattern of noise will be the same. do flip first, noise second. the image with different flips will have different noise pattern.
           Now, the code is ready, but after saving the patches with color noise, the color noise disappered after reopen.
           Okay, all set!
           Now, I begin to scale up to 2 million patches.
           
           Dr. Chen think the color noise may not be the right way. 
           
# 09-04-18 color normalization and noise
           Kernel dead: The color normalization code somehow only work in jupyter notebook. 
           Also, after extracting certain number of patches, the kernel is dead for the next sample. try to figure out what is going on.
           After discussion with Dr. Chen, we decided not adding noise and no color normalization. This is the way for their Method I.
           
# 09-05-18 patch extraction to one million
           
           This is the second step to do extract 224x224 patches. The 256x256 patches were roted 90, 180, 270, and horizontal flipped. Then 224x224 patches were randomly cropped. one 224 patch was cropped from normal patches; two 224 patches were cropped from tumor patches. So each class has one million patches. 
           The half of normal patches were generated on another computer, transfer these patches to my main computer take about 3 hours by using scp command. I will take more than 48 hours to copy to portable hard drive. 
           
           The two millions of patches were ready at the end of the day.
           
           I set up training
           
 # 09-10-18 googlenet training
 
           weird, the training model never converge. try to figure out the problem. 
           
 # 09-11-18 googlenet training
 
           Finally it begins to converge. But this is done by using batch normalization (according to the suggestion from github. lots of people have this problem). drop rate is 0.5
           
           L regulizer can be put this into lots of different layers. where should I put?
           
           So, this training has no regulizer.
           
 # 09-12-18 googlenet training
 
           2 epochs training gives me 93% accuaracy, maybe I do not need regulizer. 
           
           for the new training set, I trained model without batch normalization,  l2 regulizer is added to all the convolution layers, droprate is 0.5
           
           After meeting with Dr. chen and Kenny, the learning rate is set to reduced by half every epoch.
           
 # 09-17-18 googlenet training (continue)
           the validation score for this model is 92%.
           I did another round of training using the weight of this model. It is improved to 93%.
           
 # 09-18-18 feature extraction
           I worked on the code to extract features for random forest
           
 # 09-19-18 heatmap generation
           I worked on the code to genetrate heatmap for training dataset. 
           
           We also found that for method I, HMS&MIT group use learning rate of 0.01, 0.001, 0.0001, 0.00001, trained the model by using this learning rate.
           
           
           
