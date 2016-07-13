# cs760-machine-learning
Programming Assignments in CS760 Machine Learning (Spring 2015), by Professor David Page

This repo is for the programs I did in the Machine Learning course. There are mainly two:

1. ID3-like decision tree.
2. Bayes, which includes naive Bayes and TAN (tree-augmented naive Bayes).

The programs used Weka to make use of the Atrribute, Instance Class etc., but have not used the machine learning algorithm in Weka since the goal of these programs is to implement the algorithms. The Weka version I used is 3.7.3.

The programs should read files that are in the [ARFF](http://weka.wikispaces.com/ARFF+%28stable+version%29) format. In this format, each instance is described on a single line. The feature values are separated by commas, and the last value on each line is the class label of the instance. Each ARFF file starts with a header section describing the features and the class labels. Lines starting with '%' are comments. See the link above for a brief, but more detailed description of the ARFF format.
