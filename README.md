# White-Box Model Interpretation

Logistic Regression and Decision Tree Classifiers are being trained with the [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income) to predict weather a person makes over 50K a year. Three different models per classifier are being trained - (1.) one with the raw data set without any modifications, (2.) one with some removed features: *marital-status*, *relationship* and (3.) one with the same features removed but a balanced data set (with [xai](https://github.com/EthicalML/xai) toolbox). Then these models are being interpreted using the [eli5](https://github.com/TeamHG-Memex/eli5) library.

This project consists of one main modules: **interpret_model.py**.

- In *interpret_model.py* all models are being trained and evaluated.

## Prerequisites

- Python: 3.7.4
  - Tensorflow: 1.13.1
  - xai: 0.0.5
  - Scikit-learn: 0.20.1
  - numpy: 1.16.0
  - eli5: 0.10.1
  - pandas: 0.25.3

## Results

- **Model 1.** accuracy: 0.8108301770907974
- *Classification report*:

|Model 1.      | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.94      | 0.80   | 0.87     | 7417    |
| 1            | 0.57      | 0.85   | 0.68     | 2352    |
| micro avg    | 0.81      | 0.81   | 0.81     | 9769    |
| macro avg    | 0.76      | 0.82   | 0.77     | 9769    |
| weighted avg | 0.85      | 0.81   | 0.82     | 9769    |

- **Model 2.** accuracy: 0.7800184256321016
- *Classification report*:

| Model 2.     | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.91      | 0.78   | 0.84     | 7417    |
| 1            | 0.53      | 0.77   | 0.63     | 2352    |
| micro avg    | 0.78      | 0.78   | 0.78     | 9769    |
| macro avg    | 0.72      | 0.78   | 0.74     | 9769    |
| weighted avg | 0.82      | 0.78   | 0.79     | 9769    |

- **Model 3.** accuracy: 0.6966666666666667
- *Classification report*:

| Model 3.     | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.66      | 0.81   | 0.73     | 1200    |
| 1            | 0.75      | 0.59   | 0.66     | 1200    |
| micro avg    | 0.70      | 0.70   | 0.70     | 2400    |
| macro avg    | 0.71      | 0.70   | 0.69     | 2400    |
| weighted avg | 0.71      | 0.70   | 0.69     | 2400    |

From the classifications reports we could identify some specifics about the three models. Models 1. and 2. show almost equal performance in both accuracy and precision and recall. On the other hand, the Model 3. (balanced one) performs slightly worse than the other two (due to the smaller train data set) and shows precision and recall different results.

- [Precision](https://www.scikit-yb.org/en/latest/api/classifier/classification_report.html) is the ability of a classifier not to label an instance positive that is actually negative -> The precision percent of people that earn >= 50k is higher (less false negatives) in contrast to the other two models precision percent of people that earn < 50k is lower than the other two models.

- [Recall](https://www.scikit-yb.org/en/latest/api/classifier/classification_report.html) is the ability of a classifier to find all positive instances. All three models have almost the same recall regarding the people that < 50k (~80% of the people that earn < 50k were truly classified). But on the contrary Model 3. performs worse on the people that earn >= 50k and were truly classified (less false negatives for 1. and 2.).

- Feature importance table for each of the models. *(Weight: Feature)*

| Model 1.                                    | Model 2.                               | Model 3.                               |
|:------------------------------------------- |:-------------------------------------- |:-------------------------------------- |
| +2.318: capital-gain                        | +2.284: capital-gain                   | +2.326: capital-gain                   |
| +1.516: marital-status__ Married-civ-spouse | +1.007: occupation__ Exec-managerial   | +0.950: occupation__ Exec-managerial   |
| +1.103: relationship__ Wife                 | +0.736: occupation__ Tech-support      | +0.733: occupation__ Tech-support      |
| +0.990: marital-status__ Married-AF-spouse  | +0.722: occupation__ Prof-specialty    | +0.732: occupation__ Protective-serv   |
| +0.946: occupation__ Exec-managerial        | +0.698: education-num                  | +0.706: education-num                  |
| +0.761: education-num                       | +0.681: occupation__ Protective-serv   | +0.691: occupation__ Prof-specialty    |
| +0.723: occupation__ Tech-support           | +0.635: age                            | +0.645: age                            |
| +0.701: occupation__ Prof-specialty         | +0.556: workclass__ Federal-gov        | +0.499: workclass__ Federal-gov        |
| +0.667: workclass__ Federal-gov             | +0.465: education__ 5th-6th            | +0.450: hours-per-week                 |
| +0.657: occupation__ Protective-serv        | +0.428: occupation__ Sales             | +0.445: gender__ Male                  |
|       :    … 19 more positive …             |       :  … 18 more positive …          | +0.427: occupation__ Sales             |
|       :    … 27 more negative …             |       :  … 15 more negative …          |       :  … 19 more positive …          |
| -0.655: relationship__ Other-relative       | -0.466: ethnicity__ Amer-Indian-Eskimo |       :  … 14 more negative …          |
| -0.663: occupation__ Other-service          | -0.612: education__ Preschool          | -0.515: ethnicity__ Amer-Indian-Eskimo |
| -0.788: marital-status__ Separated          | -0.650: occupation__ Handlers-cleaners | -0.589: occupation__ Handlers-cleaners |
| -0.903: gender__ Female                     | -0.705: <BIAS>                         | -0.818: <BIAS>                         |
| -0.926: occupation__ Farming-fishing        | -0.761: occupation__ Other-service     | -0.833: occupation__ Other-service     |
| -0.932: <BIAS>                              | -0.763: occupation__ Armed-Forces      | -0.885: occupation__ Farming-fishing   |
| -1.015: marital-status__ Never-married      | -0.910: occupation__ Farming-fishing   | -1.012: workclass__ Without-pay        |
| -1.024: relationship__ Own-child            | -0.913: gender__ Female                | -1.021: education__ Preschool          |
| -1.165: workclass__ Without-pay             | -0.987: workclass__ Without-pay        | -1.263: gender__ Female                |
| -1.641: occupation__ Priv-house-serv        | -1.754: occupation__ Priv-house-serv   | -1.903: occupation__ Priv-house-serv   |

> **NOTE:** Features with largest coefficients.
> Caveats:
>
> 1. Be careful with features which are not
   independent - weights don't show their importance.
> 1. If scale of input features is different then scale of coefficients
   will also be different, making direct comparison between coefficient values
   incorrect.
> 1. Depending on regularization, rare features sometimes may have high
   coefficients; this doesn't mean they contribute much to the
   classification result for most examples.

We can now evaluate the feature importance of each model:

- On the one hand we could see that in Model 1. the columns: marital-status and relationship play an important role in deciding wheather a person should earn >= 50k. In my opinion these two columns should not be part of the models and were therefore removed in Model 2. and Model 3. Looking at Model 2. we could now see that the most important role in deciding whether a person earns >= 50k is played by the columns: *capital-gain*, *occupation* and *education-num*.

- On the other hand we see that in Model 2. a key influancer on deciding whether a persoin should earn < 50k is the gender. The feature *gender__ Female* affects the prediction in a negative way, and, therefore, I have balanced the data set in Model 3. to have equal number of *males* and *females* that earn >= 50k or < 50k (see *support* in the classification report). As we can see in the feature importance column of Model 3., this have not made any positive impact or the negativity of the *gender__ Female* feature.

Let us now interpert and evaluate some examples of the test data set. 