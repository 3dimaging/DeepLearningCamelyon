# X-Val Methods and Explanations

### K fold 
(https://www.youtube.com/watch?v=SIyMm5DFwQ8)

This video has an explanation of k-fold and code. I used this code to construct my k-fold for loop.
 - Video Code is incorrect -> "model.fit(x,y...." -> "model.fit(x_train,y_train..."
 - Otherwise, model will train on entire dataset and test on subset. Rather, we want the model to train on everything except the test set, and test on the designated test set.
