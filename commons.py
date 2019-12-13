import pandas as pd
import eli5
import enum
import xai
import logging as log
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

NUMERIC_TYPES = ["int", "float"]
RANDOM_NUMBER = 33
EXAMPLES_SPAN_ELI5 = 20
EXAMPLES_SPAN_LIME = 10
EXAMPLES_DIR_LIME = "lime_results"
TEST_SPLIT_SIZE = 0.3

# Configure logger
log.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log.getLogger().setLevel(log.DEBUG)

# Remove DataFrame display limitation
pd.set_option('display.max_columns', None)


def convert_to_lime_format(X, categorical_names, col_names=None, invert=False):
    """Converts data with categorical values as string into the right format
    for LIME, with categorical values as integers labels.
    It takes categorical_names, the same dictionary that has to be passed
    to LIME to ensure consistency.
    col_names and invert allow to rebuild the original dataFrame from
    a numpy array in LIME format to be passed to a Pipeline or sklearn
    OneHotEncoder
    """

    # If the data isn't a dataframe, we need to be able to build it
    if not isinstance(X, pd.DataFrame):
        X_lime = pd.DataFrame(X, columns=col_names)
    else:
        X_lime = X.copy()

    for k, v in categorical_names.items():
        if not invert:
            label_map = {
                str_label: int_label for int_label, str_label in enumerate(v)
            }

        else:
            label_map = {
                int_label: str_label for int_label, str_label in enumerate(v)
            }

        X_lime.iloc[:, k] = X_lime.iloc[:, k].map(label_map)

    return X_lime


def divide_features(df: pd.DataFrame) -> (list, list):
    """
    Separate the numerical from the non-numerical columns of a pandas.DataFrame.
    :param df: The pandas.DataFrame to be separated.
    :return: Two lists. One containing only the numerical column names and another one only
    the non-numerical column names.
    """
    num = []
    cat = []

    for n, t in df.dtypes.items():
        is_numeric = False
        for nt in NUMERIC_TYPES:
            if str(t).startswith(nt):
                is_numeric = True
                num.append(n)
        if not is_numeric:
            cat.append(n)

    return num, cat


class Models(enum.Enum):
    LOGISTIC_REGRESSION = 1
    DECISION_TREE = 2
    RANDOM_FOREST = 3
    XGB = 4


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
                                            random_state=RANDOM_NUMBER))])
    elif model is Models.DECISION_TREE:
        return Pipeline([("preprocessor", ct),
                         ("model", DecisionTreeClassifier(class_weight="balanced"))])
    elif model is Models.RANDOM_FOREST:
        return Pipeline([("preprocessor", ct),
                         ("model", RandomForestClassifier(class_weight="balanced", n_estimators=100, n_jobs=-1))])
    elif model is Models.XGB:
        # scale_pos_weight to make it balanced
        return Pipeline([("preprocessor", ct),
                         ("model", XGBClassifier(n_jobs=-1))])
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
                                                            test_size=TEST_SPLIT_SIZE,
                                                            random_state=RANDOM_NUMBER)
        return X_train, X_test, y_train, y_test
    else:
        raise NotImplementedError


def get_ohe_cats(model: Pipeline, cat_features: list) -> list:
    preprocessor = model.named_steps["preprocessor"]
    # Get all categorical columns (including the newly generated)
    ohe_categories = preprocessor.named_transformers_["cat"].named_steps['onehot'].categories_
    new_ohe_features = [f"{col}__{val}" for col, vals in zip(cat_features, ohe_categories) for val in vals]

    return new_ohe_features


def get_all_features(model: Pipeline, num_features: list, cat_features: list) -> list:
    return num_features + get_ohe_cats(model, cat_features)


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
