# Answers to Theoretical Questions

In this file, I report the answers from the theoretical questions from Homework 3.  
Disclaimer: this is an informal document which is meant to contain the answers to the questions in concise form.

## 1.1 TD_learning Bias
The Bellman backup is not an unbiased estimate of for $Q$. This can be easily seen from observing that the max operation used in the backup operation is not linear but convex. From this consideration it is possible to see that the Bellman backup does not preserve unbiasedness. Indee, by the application of Jensen's inequality, it can be observed that the final relation holds with inequality.


## 1.2 Tabular Learning