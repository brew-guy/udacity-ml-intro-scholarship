# Lesson 1 - Machine Learning Bird's Eye View

The big picture of ML. ML does pattern recognition. Computer science + statistics + probability models lead to more realistic understanding of real world data.

**Supervised ML:** Most used in business applications. Use labeled data to recognize unlabeled data.

**Unsupervised ML:** 2nd most used in business applications. Learning patterns from unlabeled data. Can groups items together that are likely to be similar.

**Reinforced ML:** Limited for most business use cases. Not often used with common Data Science applications. AlphaGo, autonomous vehicles, gaming agents like Open AI Gym. Learning reinforced by rewarding the model for taking certain actions.

**Classification:** Categorical outcomes. Can predict any number of outcomes from a category of choices (age, animal breed, height, color). Answers questions on the form "yes/no". E.g. a sample of red or blue points can be separated by a boundary, i.e. classified as being on one or the other side of the boundary.

**Regression:** Numeric outcomes (size, price, decimal numbers). Answers questions on the form "how much?"

**Deep Learning Neural Networks:** Beats almost all other ML algorithms in it's ability to predict. Can be used for supervised, unsupervised and reinforcement learning. High accuracy but decision making is a black box. Requires a LOT of data, a LOT of computing power. Not the focus of this program. Often predictions can be made with much simpler models that also allows to observe the decision making.

**ML tools:** TensorFlow, Scikit-learn are the world's most popular open-source libraries. Used in industry and academia.

**ML ethics:** Biases in the training data will transfer to the predictive models based on the data. Bias held by humans will inevitably transfer to computers as long as data is created by humans (video, speech, text, images). Real-world validation of models is even more important than statistical validation.

# Lesson 2 - Linear Regression

Linear models to predict quantitative values (regression). Moving a line closer to one or more points. Apply the absolute/square trick to data points to minimize error and repeat. Works best on linear data and is sensitive to outliers.

**Independant variable:** Also known as predictor. A predictor is a variable you're looking at in order to make predictions about other variables.

**Dependant variable:** Values you are trying to predict are known as dependent variables.

**Gradient descent:** A method to optimize your linear models. Draw random line, calculate error and refine until error is minimized which will lead to a good result for the linear regression problem. Can be done stochastic (one data point at a time) or in batch (multiple points at a time). In practice mini-batch gradient descent is applied (split points in batches) for speed.

**Multiple Linear Regression:** A technique for when you are comparing more than two variables. One independant variable -> 2D -> line, two independant variables -> 3D -> plane. Multiple independant variables -> higher-dimensions -> as you make a model with more predictor variables, it becomes harder to visualise, but luckily, everything else about linear regression stays the same. We can still fit models and make predictions in exactly the same way.

**Polynomial Regression:** For relationships between variables that aren't linear, where a curve or polynomial makes a better fit. The same technique is applied on a higher degree polynomial instead of a line. It requires more weights but the model solves this using gradient descent.

**Regularization:** A technique to assure that your models will not only fit to the data available, but also extend to new situations. Punishes a model for adding more complexity, i.e. when the model tries to optimize a fit with a higher degree polynomial instead of a lesser optimal fit with less complex line/polynomial. The errors and polynomial coefficients are added (_L1 regularization_) or their squares are added (_L2 regularization_) to the basic error of the model so the more complex the model, the more coefficients and higher combined error. This helps to ensure models get better at generalizing fits instead of overfitting to training data.\
_Lambda_ sets how much to punish complexity. This should be tuned depending on how much complexity to allow (e.g. medical/space travel model [low error, high complexity] vs video/SoMe recommendation [higher error, low complexity]).

**Feature Scaling:** Is used when the the prediction result will change depending on the units of the data. Especially with distance-based metrics or when incorporating regularization. Two common feature scalings are _Standardizing_, where each value is a comparison to the mean of the column, and a new, standardized value can be interpreted as the number of standard deviations the original value was from the mean. This type of feature scaling is by far the most common of all techniques. _Normalizing_, where data are scaled between 0 and 1.

# Lesson 3 - Perceptron Algorithm

A classification algorithm. Building block of neural networks. An encoding of an equation into a small graph. Takes input from nodes (features) and returns 1 or 0. They can function as logical operators.

# Lesson 4 - Decision Trees

A structure for decision making where each decision leads to a set of consequences or additional decisions. An example is an akinator (guess a person game).

**Entropy:** Entropy of a node in a decisions tree with _i_ different features.
$$entropy=-p_{1}log_{2}(p_{1})-p_{2}log_{2}(p_{2})-\cdots-p_{n}log_{2}(p_{n}) = \displaystyle\sum_{i=1}^{n}p_{i}log_{2}(p_{i})$$

**Information Gain:** The change in entropy from the parent node at a split in a decision tree. Calculated as difference between parent node entropy and the weighted average of the child nodes entropies.

**Hyperparameters:** _Maximum depth_ is simply the largest possible length between the root to a leaf. A tree of maximum depth _k_ can have at most $2^k$ leaves. _Minimum number of samples to split_ is the amount of samples the node must have in order to be large enough to split. If a node has fewer samples than _min_samples_split_ samples, it will not be split, and the splitting process stops. _Minimum number of samples per leaf_ sets a minimum for the number of samples allowed on each leaf. Avoids very uneven splits with very few samples in one leaf.

# Lesson 5 - Naive Bayes

Powerful tool for creating classifiers for incoming labeled data. Is frequently used with text data and classification problems Applies probabilistic computation in a classification task. This algorithm falls under the Supervised Machine Learning algorithm, where we can train a set of data and label them according to their categories. n the heart of Naive Bayes algorithm is the probabilistic model that computes the conditional probabilities of the input features and assigns the probability distributions to each of possible classes.

**Naive assumption:** In Naive Bayes assumes that probabilities are independant, e.g. can't use features that relate to each other (like probability of outside temperature being hot/cold at the same time).

**Bayes theorem:** Infers (posterior) knowledge from known (prior) probabilites when an event occurs.
$$P(A|R)=\frac{P(A)P(R|A)}{P(A)P(R|A)+P(B)P(R|B)}$$

**Sensitivity:** (True Positive Rate) refers to the probability of a positive test, conditioned on truly having the condition (or tested positive by the `Gold Standard test` if the true condition can not be known).

**Specificity:** (True Negative Rate) refers to the probability of a negative test, provided one does not have the condition (judged negative by the `Gold Standard`).

**Bayes Nets:** If the output of the Naive Bayes algorithm is a classification, the output for the Bayes Net is a probability distribution. Furthermore, while the Naive Bayes assumes conditional independence, the more general Bayes Nets specify the attributes in probability distributions and conditional independence.

# Lesson 6 - Support Vector Machines

Powerful algorithm for classification that both classifies the data but also aims to find the best possible boundary (that maintains the largest distance from the points).

**Margin Error:** Punishes models that have smaller margins, i.e. large margin -> small error, small margin -> large error.

**SVM Error Function:** In SVMs, the error function is the sum of the classification error and the margin error. This is the error that the model will seek to minimize with gradient decent.

**C Parameter:** (A parallel to regularization lambda?) A hyperparameter and constant that is multiplied to the classification error in the Error Function (classification error + margin error).
This should be tuned depending on how much weight to shift to the classification error in the error function, thereby shifting towards accurate classification rather than finding a good margin. A small C -> large margin + allow for classification errors, large C -> accurate classification but small margin.

**Maximum Margin Classifier:** When the data can be completely separated, the linear version of SVMs attempts to maximize the distance from the linear boundary to the closest points (called the support vectors).

**Classification with Inseparable Classes:** Unfortunately, data in the real world is rarely completely separable. For this reason is the hyper-parameter C. The C hyper-parameter determines how flexible we are willing to be with the points that fall on the wrong side of our dividing boundary. The value of C ranges between 0 and infinity. When _C_ is large, you are forcing your boundary to have fewer errors than when it is a small value. _Note:_ when _C_ is too large for a particular set of data, you might not get convergence at all because your data cannot be separated with the small number of errors allotted with such a large value of _C_.

**Polynomial kernel:** The Kernel Trick. Highly effective classification by use of polynomials instead of lines. The degree of polynomial to use is a hyperparameter.

**RBF Kernel:** Radial Basis Functions Kernel. The RBF kernel allows the opportunity to classify points that seem hard to separate in any space. This is a density based approach that looks at the closeness of points to one another. Applies a "mountain" / radial basis function to each point to help classify points in a higher dimension by boundaries. The _gamma parameter_ is a hyperparameter that sets the wideness/narrowness of the "mountain". When _gamma_ is large, the outcome is similar to having a large value of _C_, that is your algorithm will attempt to classify every point correctly. Alternatively, small values of _gamma_ will try to cluster in a more general way that will make more mistakes, but may perform better when it sees new data.

# Lesson 7 - Ensemble Methods

Combines results from multiple models (weak learners) e.g. by averaging, voting etc. The resulting model is called strong learner. The weak learners are not required to be good, just slightly better than random chance. The weak learners for ensemble methods are often decision trees. By combining algorithms, we can often build models that perform better by meeting in the middle in terms of bias and variance. These ideas are based on minimizing bias and variance based on mathematical theories, like the _central limit theorem_. Another method that is used to improve ensemble methods is to introduce randomness into high variance algorithms before they are ensembled together.

**Bagging:** **B**ootstrap **agg**regat**ing**.

**Boosting:** E.g. AdaBoost (adaptive boosting). Punishes more for misclassified values on each run of the weak learners. Ensemble learners together in a way that allows those that perform best in certain areas to create the largest impact.

**Bias:** When a model has high bias, this means that means it doesn't do a good job of bending to the data. An example of an algorithm that usually has high bias is linear regression. Even with completely different datasets, we end up with the same line fit to the data. When models have high bias, this is bad.

**Variance:** When a model has high variance, this means that it changes drastically to meet the needs of every point in our dataset. Linear models has low variance, but high bias. An example of an algorithm that tends to have high variance and low bias is a decision tree (especially decision trees with no early stopping parameters). A decision tree, as a high variance algorithm, will attempt to split every point into its own branch if possible. This is a trait of high variance, low bias algorithms - they are extremely flexible to fit exactly whatever data they see.

**Randomness:** The introduction of randomness combats the tendency of these algorithms to overfit (or fit directly to the data available). There are two main ways that randomness is introduced: _Bootstrap the data_ - that is, sampling the data with replacement and fitting your algorithm to the sampled data. _Subset the features_ - in each split of a decision tree or with each algorithm used in an ensemble, only a subset of the total possible features are used. These are the two random components used in the algorithm called _random forests_.

**Random Forests:** Decision trees tend to overfit a lot. A solution is to pick a subset of features randomly several times and base the prediction on a vote from the results.

## General process for training models

In general, there is a five step process that can be used each time you want to use a supervised learning method (which you actually used above):

- Import the model
- Instantiate the model with the hyperparameters of interest
- Fit the model to the training data
- Predict on the test data
- Score the model by comparing the predictions to the actual values

# Lesson 8 - Model Evaluation Metrics

The main metrics to evaluate models.

**Testing:** THOU SHALT NEVER USE YOUR TESTING DATA FOR TRAINING.

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
```

## Classification metrics

**Confusion matrix:**

- Type 1 Error (Error of the first kind, or False Positive): In the medical example, this is when we misdiagnose a healthy patient as sick.
- Type 2 Error (Error of the second kind, or False Negative): In the medical example, this is when we misdiagnose a sick patient as healthy.

|                 | Predicted positive | Predicted negative |
| --------------- | :----------------: | -----------------: |
| Actual positive |   True positive    |     False negative |
| Actual negative |   False positive   |      True negative |

**Accuracy:** 0-1. Tells how well the model classified correctly. The ratio of correctly classified samples per total samples = $\frac{TP+TN}{TP+TN+FP+FN}$. Accuracy is not necessarily a good measure alone. If the data is skewed, the model with high accuracy can still miss all the samples it should be able to classify. E.g. credit card fraudulent/correct transaction samples that are distributed 0,01% / 99,99% will not catch fraudulent samples even with an accuracy of 99%.

**Precision:** 0-1. What proportion of positive predictions was actually correct? Predicted positive ratio = $\frac{TP}{TP+FP}$. With high precision, the model will not yield any false positives. Good for spam classification where we can live with a few spam messages slipping into the inbox but not with non-spam being lost in the spam folder.

**Recall (sensitivity):** 0-1. What proportion of actual positives was predicted correctly? Actual positive ratio = $\frac{TP}{TP+FN}$. With high recall, the model will not yield any false negatives. Good for medical models where it's just inconvenient for healthy patients to be incorrectly classified as ill but potentially disastrous for ill patients to be sent home due to being classified as healthy.

**Specificity:** Actual negative ratio = $\frac{TN}{TN+FP}$.

**F1 Score:** 0-1. The _Harmonic Mean_ of the Precision and Recall scores. $F_1 = \frac{2*Precision*Recall}{Precision+Recall}$ which is always lower than the arithmetic mean (average), and only close to the arithmetic mean when the two variables are close to being equal. So if Precision or Recall is particularly low, the F1 Score raises a flag as it will also be lower than the arithmetic average.

**F-beta Score:** 0-1. Skewers the score more towards Precision or Recall. The smaller the beta, the more towards Precision. F0.25 score is skewered more towards Precision. F2 score puts more weight on Recall. $F_\beta = (1 + \beta^2)\frac{Precision*Recall}{\beta^2*Precision+Recall}$

**ROC Curve:** Receiver Operating Characteristic Curve. An Area Under Curve (AUC) relation to how well a model is able to split a sample when plotting the (FP-rate, TP-rate) in a plot. $FPR=\frac{FP}{FP+TN}$, $TPR=\frac{TP}{TP+FN}$. A perfect split will yield AUC=1, a good split AUG=0.8, random/bad split AUC=0.5. _Does the ROC Curve indicate easily splittable data or is it more an indicator of whether the model is good?_

## Regression metrics

**Mean Absolute Error:** The average of absolute values of the distance from points to the fit model. Useful metric to optimize on when the value you are trying to predict follows a skewed distribution. Optimizing on an absolute value is particularly helpful in these cases because outliers will not influence models attempting to optimize on this metric as much as if you use the mean squared error. The optimal value for this technique is the median value. When you optimize for the R2 value of the mean squared error, the optimal value is actually the mean.

**Mean Squared Error:** Same as Mean Absolute Error but with average of squared distances.

**R2 Score:** 0 (bad) to 1 (good model). Uses the simplest possible model that fits the data (e.g. an average line through points), finds the ratio between the actual model and simple model to calulate R2. $R2=1-\frac{Mean\:Squared\:Error}{Simple\:Model\:Error}$

# Lesson 9 - Training and Tuning
