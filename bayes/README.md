## Naive bayes and TAN

The goal of this assignment is to implement both naive Bayes and TAN (tree-augmented naive Bayes).

The program should be called as follows, where the last argument is a single character (either 'n' or 't') that indicates whether to use naive Bayes or TAN:
```
bayes <train-set-file> <test-set-file> <n|t>
```

The program has the following assumptions:

* The code is intended for binary classification problems.
* All of the variables are discrete valued. The program is able to handle an arbitrary number of variables with possibly different numbers of values for each variable.
* Laplace estimates (pseudocounts of 1) is used when estimating all probabilities.

For the TAN algorithm, the program:

* Use Prim's algorithm to find a maximal spanning tree (but choose maximal weight edges instead of minimal weight ones). Initialize this process by choosing the first variable in the input file for Vnew. If there are ties in selecting minimum weight edges, use the following preference criteria: (1) prefer edges emanating from variables listed earlier in the input file, (2) if there are multiple minimal weight edges emanating from the first such variable, prefer edges going to variables listed earlier in the input file.
* To root the maximal weight spanning tree, pick the first variable in the input file as the root.
