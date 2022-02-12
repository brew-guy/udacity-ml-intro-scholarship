## Recap

In this lesson, you learned about a number of techniques used in ensemble methods. Before looking at the techniques, you saw that there are two variables with tradeoffs **Bias** and **Variance**.

- **High Bias, Low Variance** models tend to underfit data, as they are not flexible. **Linear models** fall into this category of models.

- **High Variance**, Low Bias models tend to overfit data, as they are too flexible. **Decision trees** fall into this category of models.

## Ensemble Models

In order to find a way to optimize for both variance and bias, we have ensemble methods. Ensemble methods have become some of the most popular methods used to compete in competitions on Kaggle and used in industry across applications.

There were two randomization techniques you saw to combat overfitting:

1. **Bootstrap the data** - that is, sampling the data with replacement and fitting your algorithm and fitting your algorithm to the sampled data.

2. **Subset the features** - in each split of a decision tree or with each algorithm used an ensemble only a subset of the total possible features are used.

## Techniques

You saw a number of ensemble methods in this lesson including:

- [BaggingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier)
- [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
- [AdaBoostClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)
  Another really useful guide for ensemble methods can be found [in the documentation here](http://scikit-learn.org/stable/modules/ensemble.html). These methods can also all be extended to regression problems, not just classification.

## Additional Resources

Additionally, here are some great resources on AdaBoost if you'd like to learn some more!

- Here is the original [paper](https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf) from Freund and Schapire.
- A follow-up [paper](https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf) from the same authors regarding several experiments with Adaboost.
- A great [tutorial](http://rob.schapire.net/papers/explaining-adaboost.pdf) by Schapire.
