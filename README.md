# A Simple Artifical Neural Network implemented in C
  



# Introduction

Artificial Neural Network (ANN) is a non-linear classification algorithm that is used in many machine learning
A type of ANN is a multilayer perceptron which is a feed-forward network that maps the input to its corresponding
output.

The Backpropagation Algorithm is a method of training a multilayer perceptron network by minimizing the appropriate
cost function of its output.

In this study, the Backpropagtion algorithm will be implemented to train the multilayer perceptron on the given
dataset. The performance of the classifier will also be compared to the SVM classifier.

# Methodology

## Data Set
Table 1 shows the data files given. The training set contains 3486 instances, with 354
attributes. The training set is labeled which can be found in `data_labels.csv`. The class
labels are 1,2,3,4,5,6,7,8. The test set contains 701 instances.


Table: Summary of data files given

| Filename         | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| data.csv         | 3486 instances each having 354 attributes or features          |
| data_labels.csv  | class labels (1,2,3,4,5,6,7,8) for each of the 3486 instances  |
| test_set.csv     | 701 test instances without labels                              |

Table 2 shows the number of instances per class. As can be seen from the `Count` column
the training data set is highly imbalanced especially in classes 1, 3, and 7. To deal with
this, classes 3 and 7 are augmented by repeating the instances 8x and 5x respectively.
The resulting number of instances is shown in the `Count After Augmentation` column. The
training set is then divided into training set and validation set with a 2/3 to 1/3 ratio.


Table: Summary of Training Data and Data Augmentation Procedure

 Class   Count  Data Augmentation    Count After Augmentation   # Training   # Validation
------  ------  ------------------  -------------------------  -----------  -------------
     1    1625  none                                     1625         1084            541
     2     233  none                                      233          156             77
     3      30  repeated 8x                               240          160             80
     4     483  none                                      483          322            161
     5     287  none                                      287          192             95
     6     310  none                                      310          207            103
     7      52  repeated 5x                               260          174             86
     8     466  none                                      466          311            155

## ANN Program
The Multilayer Perceptron Network with the Backpropagation Algorithm was implemented in C.
The network contains an input layer with 354 nodes which corresponds to features, an output
layer with 8 nodes which corresponds to the classes, and 2 hidden layers.

The ANN program can be run in the terminal by typing:
```
./ann <no of nodes in h1> <no of nodes in h2> <maximum epoch> <learning rate> 
      <f1 score threshold> <epoch to print temp results>
```
For example `./ann 50 50 3000 0.5 0.99 10` will run the ANN program with 50 nodes in hidden layer 1 and 2 for 3000 epochs with a 0.5 learning rate, a 0.99 f1 score threshold, and will print temporary results every 10 epoch. For this particular settings the ANN program will 
end its run if it already reached 3000 epochs or if the average f1 score for the 8 classes
reaches 0.99. The temporary results that will be printed in the terminal contains the
confusion matrix, the precision, recall, and f1 scores for each class and the overall
error rate. A sample can be seen below.

```
epoch no. 170
TRAINING RESULTS:
confusion matrix
[1070]  0001   0000   0000   0000   0007   0002   0004  
 0002  [0146]  0000   0000   0000   0006   0000   0002  
 0000   0000  [0160]  0000   0000   0000   0000   0000  
 0000   0000   0000  [0312]  0000   0002   0000   0008  
 0000   0000   0000   0000  [0192]  0000   0000   0000  
 0001   0001   0001   0001   0000  [0184]  0007   0012  
 0001   0000   0000   0000   0000   0004  [0169]  0000  
 0002   0002   0000   0009   0000   0009   0001  [0288] 
tp: 1070, 146, 160, 312, 192, 184, 169, 288, 
fp: 6, 4, 1, 10, 0, 28, 10, 26, 
fn: 14, 10, 0, 10, 0, 23, 5, 23, 
precision: 0.99, 0.97, 0.99, 0.97, 1.00, 0.87, 0.94, 0.92, 
recall: 0.99, 0.94, 1.00, 0.97, 1.00, 0.89, 0.97, 0.93, 
f1 score: 0.99, 0.95, 1.00, 0.97, 1.00, 0.88, 0.96, 0.92, 
f1 score mean: 0.96
error rate: 0.0326170
```

Table 3 shows the summary of the different runs made using the ANN impementation on the data.
The ANN program was run using 2 sets of hidden layer nodes: 50 and 100 and 3 sets of learning
rate: 0.5, 0.1, and 0.01. The number of nodes selected for each run was chosen to be less
than the number of features to avoid overfitting the network. 


Table: Network specifications for the different ANN Runs

 Run   Nodes HL1   Nodes HL2   Learning Rate
----  ----------  ----------  --------------
   1          50          50            0.50
   2          50          50            0.10
   3          50          50            0.01
   4         100         100            0.50
   5         100         100            0.10
   6         100         100            0.01

## Support Vector Machine (SVM) as classifier
The ANN implementation was also compared to the Support Vector Machine classifier.
The `svm` function under the `e1071` R-package was used to train the classifier
to the training data. The `predict` function was used to get the results from the
trained model.

```
svm_learn <- svm(Y~.,data=train.data,type="C-classification",kernel="linear")
pred.valid <- predict(svm_learn,newdata=valid.data)
```

# Results

## Results of ANN Training and Validation
Table 4 shows the summary of the results for the different runs made in the study.
Figures 1 - 6 shows the graphs of precision, recall, f1 score, and error rate for each of the
runs made in the study. The graphs also compares the result between the training and
validation sets.
It can be seen from Table 4 that regardless of the number of hidden nodes used in the network,
0.5 and 0.01 learning rates are much slower to converge than using 0.1 learning rate.
Runs 2 and 5 have learning rates of 0.1 and the number of epochs that the network
reached a mean f1 score of 0.99 were 560 and 630 respectively. This can be explained
by looking at Figures 1, 3, 4, and 6. Figures 1 and 4 show that the learning rate
0.5 is large enough to miss the maxima (0.99 f1 score) many times that it took the
network longer to converge as shown by the noisier lines in the graphs. Figures 3 and
6, on the other hand, show that the learning rate 0.01 is too small such that it also
took the network to converge longer than by using the learning of 0.1.




![Result of Run 1 - HL1: 50, HL2: 50, LR: 0.5](https://github.com/clutariomark/neuralnetwork/raw/master/images/h50h50l50-1.png)

![Result of Run 2 - HL1: 50, HL2: 50, LR: 0.1](https://github.com/clutariomark/neuralnetwork/raw/master/images/h50h50l10-1.png)

![Result of Run 3 - HL1: 50, HL2: 50, LR: 0.01](https://github.com/clutariomark/neuralnetwork/raw/master/images/h50h50l1-1.png)

![Result of Run 4 - HL1: 100, HL2: 100, LR: 0.5](https://github.com/clutariomark/neuralnetwork/raw/master/images/h100h100l50-1.png)

![Result of Run 5 - HL1: 100, HL2: 100, LR: 0.1](https://github.com/clutariomark/neuralnetwork/raw/master/images/h100h100l10-1.png)

![Result of Run 6 - HL1: 100, HL2: 100, LR: 0.01](https://github.com/clutariomark/neuralnetwork/raw/master/images/h100h100l1-1.png)


Table: Summary of the different runs

 run   epoch    precison      recall    f1 score
----  ------  ----------  ----------  ----------
   1    2072   0.9896254   0.9908292   0.9902109
   2     567   0.9879322   0.9921651   0.9900266
   3    2559   0.9894935   0.9907600   0.9900859
   4    2376   0.9884298   0.9919810   0.9901580
   5     630   0.9893539   0.9916270   0.9904760
   6    2492   0.9893582   0.9916616   0.9904861

\pagebreak

## Comparison of ANN and SVM
Table 7 shows the comparison of ANN and SVM classifiction results on the Validation Set.
It can be seen that performance-wise, SVM and ANN are considered the same. SVM has an mean
f1 score of about 0.95 while the ANN runs resulted to scores ranging from 0.95 to 0.97.
However, in terms of processing time SVM is better since it only took 16.76 seconds to train 
the classifier on the training data while it took 2 minutes and 37 seconds
to train the ANN with a learning rate of 0.1.


Table: Summary of SVM Classification Result (Validation Set)

 class    tp   fp   fn    accuracy   precision      recall     f1score
------  ----  ---  ---  ----------  ----------  ----------  ----------
     1   527    7   13   0.9845798   0.9868914   0.9759259   0.9813780
     2    74   10    3   0.9899769   0.8809524   0.9610390   0.9192547
     3    80    1    0   0.9992290   0.9876543   1.0000000   0.9937888
     4   161    1    0   0.9992290   0.9938272   1.0000000   0.9969040
     5    95    1    0   0.9992290   0.9895833   1.0000000   0.9947644
     6    97   11    6   0.9868928   0.8981481   0.9417476   0.9194313
     7    86    1    0   0.9992290   0.9885057   1.0000000   0.9942197
     8   131   14   24   0.9707016   0.9034483   0.8451613   0.8733333



Table: Summary of SVM and ANN classification on Validation Set (mean)

run    precision      recall     f1score
----  ----------  ----------  ----------
SVM    0.9536263   0.9654842   0.9591343
1      0.9540532   0.9561211   0.9546794
2      0.9659724   0.9668601   0.9661641
3      0.9681576   0.9704982   0.9692056
4      0.9546368   0.9638578   0.9582992
5      0.9586308   0.9653948   0.9613470
6      0.9713116   0.9757014   0.9733942

# Conclusions
Based on the results obtained in the study, ANN can be a good classifier since its performance
the same and sometimes can be even better than known strong classifiers such as SVM. However,
it should be noted that the processing time is much longer than the SVM.
