Prediction Assignment Writeup
-----------------------------

Raman Chandrasekar
May 1, 2016



## Overall Approach

When I tried out some approaches on the training and testing data, I found that pretty much any approach takes times on my 2010 MacBook Pro, even with 8GB of memory. I found the article [Improving Performance of Random Forest in caret::train()](https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md) by Len Greski very useful,  primarily in the making the processing parallel.

From previous experience with machine learning, I have become fond of using Random Forest (rf) for prediction.  I thought I would try rf and a few other methods on the training data as supplied, to get a baseline. But it turned out that methods such as boosting failed on the training and/or testing data as supplied.

I was able to get a RF model with the training data using the following method. As recommended in the Greski article, I set up my laptop as a 3-core cluster (leaving 1 core for the OS). I set up traincontrol to do 10-fold cross validation, and set it up to use the cluster defined. I then loaded the training data into a dataframe, and trained the **classe** feature/variable against all other features/variables. It took quite a while but finally ended. I then stopped the cluster. The results from this run are presented in the next section.


## Results from the baseline Random Forest training

When I examined the model, I got the following:

```Random Forest 
 
19622 samples
 159 predictor
 5 classes: 'A', 'B', 'C', 'D', 'E' 
 
 No pre-processing
 Resampling: Cross-Validated (10 fold) 
 Summary of sample sizes: 365, 365, 366, 367, 365, 366, ... 
 Resampling results across tuning parameters:
         
         mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
 2  0.2684459  0.0000000  0.005065499  0.00000000
 117  0.9067042  0.8819656  0.048654401  0.06169690
 6952  0.9926829  0.9907819  0.011781607  0.01484254
```

Clearly an Accuracy of 0.993 is very good. In fact it is so good, that we need to worry about possibly over-fitting. The next command gave an idea of how each of the 10-fold cross validation steps performed.  The average of these is the 0.993 accuracy we got earlier.

```
fit$resample
 Accuracy     Kappa Resample
 1  1.0000000 1.0000000   Fold01
 2  1.0000000 1.0000000   Fold02
 3  0.9756098 0.9692423   Fold05
 4  1.0000000 1.0000000   Fold04
 5  1.0000000 1.0000000   Fold03
 6  1.0000000 1.0000000   Fold06
 7  1.0000000 1.0000000   Fold09
 8  1.0000000 1.0000000   Fold08
 9  0.9756098 0.9692884   Fold07
 10 0.9756098 0.9692884   Fold10
```

The next command showed that the prediction did very well except for a small confusion between classes D and E.
```
confusionMatrix.train(fit)
Cross-Validated (10 fold) Confusion Matrix 

(entries are percentages of table totals)
 
          Reference
Prediction    A    B    C    D    E
         A 26.8  0.0  0.0  0.0  0.0
         B  0.0 19.2  0.2  0.0  0.0
         C  0.0  0.2 17.0  0.0  0.0
         D  0.0  0.0  0.0 17.0  0.2
         E  0.0  0.0  0.0  0.0 19.2
```

However, there were all sorts of errors when I tried to predict using this model, with data errors etc. However, since the accuracy was really high, I decided to stick with this RF approach, but to clean up the data and try again.

## Cleaning up the data

On examining the data, it is clear that there are many columns with NA values. I wrote a quick R function to identify columns with over 60% NAs in the training data, and remove these columns from  both the training and testing data. I also decided that user_name and any timestamp data should not be factor in deciding the results. I removed these columns as well. From a total of 160 columns, this took me to 89 columns each, with 19,622 rows for training and 20 for testing. 

The training and testing data differ in some respects. The training data has the **classe** column, while the testing data has the **problem\_id** column.  I sorted the column names in both the training and testing datasets, and put **classe** and **problem_id** at the end of these respectively. This sorting is more for ease of manual inspection, if required.

I reran RandomForest on this reduced dataset.

## Results on running RandomForest on reduced data



