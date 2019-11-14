import pandas as pd
import eli5
import enum
import xai
import random as rd
import logging as log
import commons
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xai import data

# Configure logger
log.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log.getLogger().setLevel(log.DEBUG)

# Remove DataFrame display limitation
pd.set_option('display.max_columns', None)

TARGET = "loan"
REMOVE_FEATURES_M2 = ["marital-status", "relationship"]
REMOVE_FEATURES_M4 = ["gender"]


class Models(enum.Enum):
    LOGISTIC_REGRESSION = 1
    DECISION_TREE = 2


class Splits(enum.Enum):
    IMBALANCED = 1
    BALANCED__GENDER = 2
    BALANCED__ETHNICITY = 3


def get_column_transformer(numerical: list, categorical: list) -> ColumnTransformer:

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    return ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical),
                    ('cat', categorical_transformer, categorical)])


def get_pipeline(ct: ColumnTransformer, model: Models) -> Pipeline:

    if model is Models.LOGISTIC_REGRESSION:
        return Pipeline([("preprocessor", ct),
                         ("model",
                         LogisticRegression(class_weight="balanced",
                                            solver="liblinear",
                                            random_state=commons.RANDOM_NUMBER))])
    elif model is Models.DECISION_TREE:
        return Pipeline([("preprocessor", ct),
                         ("model", DecisionTreeClassifier(class_weight="balanced"))])
    else:
        raise NotImplementedError


def get_split(split: Splits, cat_features: list, df_x: pd.DataFrame, df_y: pd.Series)\
        -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):

    balance_on = ""
    if split is Splits.BALANCED__GENDER or split is Splits.BALANCED__ETHNICITY:
        if split is Splits.BALANCED__GENDER:
            balance_on = "gender"
        elif split is Splits.BALANCED__ETHNICITY:
            balance_on = "ethnicity"
        X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, train_idx, test_idx = \
            xai.balanced_train_test_split(
                df_x, df_y, balance_on,
                min_per_group=600,
                max_per_group=600,
                categorical_cols=cat_features)
        return X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced
    elif split is Splits.IMBALANCED:
        X_train, X_test, y_train, y_test = train_test_split(df_x,
                                                            df_y,
                                                            stratify=df_y,
                                                            test_size=.3,
                                                            random_state=commons.RANDOM_NUMBER)
        return X_train, X_test, y_train, y_test
    else:
        raise NotImplementedError


def get_all_features(model: Pipeline, num_features: list, cat_features: list) -> list:
    preprocessor = model.named_steps["preprocessor"]
    # Get all categorical columns (including the newly generated)
    ohe_categories = preprocessor.named_transformers_["cat"].named_steps['onehot'].categories_
    new_ohe_features = [f"{col}__{val}" for col, vals in zip(cat_features, ohe_categories) for val in vals]
    # Append the all numerical and categorical features
    return num_features + new_ohe_features


def train_model(model_type: Models, split: Splits, df_x: pd.DataFrame, df_y: pd.Series,
                num_features: list, cat_features: list):

    log.debug("Numerical features: {}".format(num_features))
    log.debug("Categorical features: {}".format(cat_features))

    # Transform the categorical features to numerical
    preprocessor = get_column_transformer(num_features, cat_features)

    model = get_pipeline(preprocessor, model_type)

    X_train, X_test, y_train, y_test = get_split(split, cat_features, df_x, df_y)

    # Now we can fit the model on the whole training set and calculate accuracy on the test set.
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)
    log.info("Model accuracy: {}".format(accuracy_score(y_test, y_pred)))
    log.info("Classification report: \n{}".format(classification_report(y_test, y_pred)))

    return model


def interpret_model(model: Pipeline, all_features: list):
    log.debug("All features: {}".format(all_features))

    # Explain the model
    log.info(
        eli5.format_as_text(
            eli5.explain_weights(
                model.named_steps["model"],
                feature_names=all_features)))


def print_results(models: [Pipeline], all_features1: list, all_features2: list, all_features4: list,
                  df_X: pd.DataFrame, df_y: pd.Series):

    _, X_test1, _, y_test1 = train_test_split(df_X,
                                              df_y,
                                              stratify=df_y,
                                              test_size=.3,
                                              random_state=commons.RANDOM_NUMBER)

    X_test2 = X_test1.drop(REMOVE_FEATURES_M2, axis=1)
    X_test4 = X_test2.drop(REMOVE_FEATURES_M4, axis=1)

    for i in range(20):
        number = rd.randrange(0, len(X_test1))

        # Model 1
        model_all = models[0]
        explain_prediction(model_all, X_test1, y_test1, all_features1, number)

        # Model 2
        model_par = models[1]
        explain_prediction(model_par, X_test2, y_test1, all_features2, number)

        # Model 3
        model_par_bal = models[2]
        explain_prediction(model_par_bal, X_test2, y_test1, all_features2, number)

        # Model 4
        model_par_bal = models[3]
        explain_prediction(model_par_bal, X_test4, y_test1, all_features4, number)

        log.debug("\n**********************************************************************************************\n")


def explain_prediction(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, all_features: list, number: int):
    log.debug("\n{}".format(X_test.iloc[number]))
    log.info(
        eli5.format_as_text(
            eli5.explain_prediction(
                model.named_steps["model"],
                model.named_steps["preprocessor"].transform(X_test)[number],
                feature_names=all_features)
        )
    )
    log.debug("Result: {}\n".format(y_test.iloc[number]))


# Prepare the data
df = data.load_census()
df_X1 = df.drop(TARGET, axis=1)
# Model 2 and 3 use the same columns
df_X2 = df_X1.drop(REMOVE_FEATURES_M2, axis=1)
df_X4 = df_X2.drop(REMOVE_FEATURES_M4, axis=1)
df_y = df[TARGET].map({" <=50K": 0, " >50K": 1})

log.debug("\n{}".format(df_y.value_counts()))

num_features1, cat_features1 = commons.divide_features(df_X1)
num_features2, cat_features2 = commons.divide_features(df_X2)
num_features4, cat_features4 = commons.divide_features(df_X4)

# All features included.
lr_model1 = train_model(Models.LOGISTIC_REGRESSION, Splits.IMBALANCED, df_X1, df_y, num_features1, cat_features1)
interpret_model(lr_model1, get_all_features(lr_model1, num_features1, cat_features1))
log.info("\n*******************************************************************************************************\n")

# Some features were removed.
lr_model2 = train_model(Models.LOGISTIC_REGRESSION, Splits.IMBALANCED, df_X2, df_y, num_features2, cat_features2)
interpret_model(lr_model2, get_all_features(lr_model2, num_features2, cat_features2))
log.info("\n*******************************************************************************************************\n")

# Removed features + balanced dataset.
lr_model3 = train_model(Models.LOGISTIC_REGRESSION, Splits.BALANCED__GENDER, df_X2, df_y, num_features2, cat_features2)
interpret_model(lr_model3, get_all_features(lr_model3, num_features2, cat_features2))
log.info("\n*******************************************************************************************************\n")

# Same as 3 with more features removed.
lr_model4 = train_model(Models.LOGISTIC_REGRESSION, Splits.BALANCED__ETHNICITY,
                        df_X4, df_y, num_features4, cat_features4)
interpret_model(lr_model4, get_all_features(lr_model4, num_features4, cat_features4))
log.info("\n*******************************************************************************************************\n")

# Print results from examples of the test data set.
print_results([lr_model1, lr_model2, lr_model3, lr_model4],
              get_all_features(lr_model1, num_features1, cat_features1),
              get_all_features(lr_model2, num_features2, cat_features2),
              get_all_features(lr_model4, num_features4, cat_features4),
              df_X1,
              df_y)

# Decision Tree Models
# dt_model1 = train_model(Models.DECISION_TREE, df_X1, df_y, num_features1, cat_features1)
# interpret_model(dt_model1, get_all_features(dt_model1, num_features1, cat_features1))

# dt_model2 = train_model(Models.DECISION_TREE, df_X2, df_y, num_features2, cat_features2)
# interpret_model(dt_model2, get_all_features(dt_model2, num_features2, cat_features2))
