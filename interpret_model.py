import os
import pandas as pd
import eli5
import random as rd
import logging as log
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xai import data
from interpret import commons
from lime.lime_tabular import LimeTabularExplainer
from functools import partial
from datetime import datetime


# Configure logger
log.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log.getLogger().setLevel(log.DEBUG)

# Remove DataFrame display limitation
pd.set_option('display.max_columns', None)

TARGET = "loan"
REMOVE_FEATURES_M2 = ["marital-status", "relationship"]
REMOVE_FEATURES_M4 = ["gender"]
REMOVE_FEATURES_M5 = ["ethnicity"]


def print_eli5_local_results(model: Pipeline, df_X: pd.DataFrame, df_y: pd.Series, all_features: list, span: int):

    _, X_test, _, y_test = train_test_split(df_X,
                                            df_y,
                                            stratify=df_y,
                                            test_size=commons.TEST_SPLIT_SIZE,
                                            random_state=commons.RANDOM_NUMBER)

    for i in range(span):
        random_number = rd.randrange(0, len(X_test))

        log.debug("\n{}".format(X_test.iloc[random_number]))
        log.info(
            eli5.format_as_text(
                eli5.explain_prediction(
                    model.named_steps["model"],
                    model.named_steps["preprocessor"].transform(X_test)[random_number],
                    feature_names=all_features)
            )
        )
        log.debug("Result: {}\n".format(y_test.iloc[random_number]))
        log.debug("\n**********************************************************************************************\n")


def interpret_lr_models():
    # All features included.
    lr_model1 = commons.train_model(commons.Models.LOGISTIC_REGRESSION, commons.Splits.IMBALANCED, df_X1, df_y,
                                            num_features1, cat_features1)
    commons.interpret_model(lr_model1, commons.get_all_features(lr_model1, num_features1, cat_features1))

    # Print results from examples of the test data set.
    print_eli5_local_results(lr_model1,
                             df_X1,
                             df_y,
                             commons.get_all_features(lr_model1, num_features1, cat_features1),
                             commons.EXAMPLES_SPAN_ELI5)
    log.info(
        "\n*******************************************************************************************************\n")

    # Some features were removed.
    lr_model2 = commons.train_model(commons.Models.LOGISTIC_REGRESSION, commons.Splits.IMBALANCED, df_X2, df_y,
                                    num_features2, cat_features2)
    commons.interpret_model(lr_model2, commons.get_all_features(lr_model2, num_features2, cat_features2))

    # Print results from examples of the test data set.
    print_eli5_local_results(lr_model2,
                             df_X2,
                             df_y,
                             commons.get_all_features(lr_model2, num_features2, cat_features2),
                             commons.EXAMPLES_SPAN_ELI5)
    log.info(
        "\n*******************************************************************************************************\n")

    # Removed features + balanced dataset.
    lr_model3 = commons.train_model(commons.Models.LOGISTIC_REGRESSION, commons.Splits.BALANCED__GENDER, df_X2, df_y,
                                    num_features2, cat_features2)
    commons.interpret_model(lr_model3, commons.get_all_features(lr_model3, num_features2, cat_features2))

    # Print results from examples of the test data set.
    print_eli5_local_results(lr_model3,
                             df_X2,
                             df_y,
                             commons.get_all_features(lr_model2, num_features2, cat_features2),
                             commons.EXAMPLES_SPAN_ELI5)
    log.info(
        "\n*******************************************************************************************************\n")

    # Same as 3 with more features removed.
    lr_model4 = commons.train_model(commons.Models.LOGISTIC_REGRESSION, commons.Splits.BALANCED__ETHNICITY,
                                    df_X4, df_y, num_features4, cat_features4)
    commons.interpret_model(lr_model4, commons.get_all_features(lr_model4, num_features4, cat_features4))

    # Print results from examples of the test data set.
    print_eli5_local_results(lr_model4,
                             df_X4,
                             df_y,
                             commons.get_all_features(lr_model4, num_features4, cat_features4),
                             commons.EXAMPLES_SPAN_ELI5)
    log.info(
        "\n*******************************************************************************************************\n")


def interpret_dt_models():
    """
    TODO: Implement function.
    :return:
    """

    """
    # Decision Tree Models
    dt_model1 = commons.train_model(commons.Models.DECISION_TREE, df_X1, df_y, num_features1, cat_features1)
    commons.interpret_model(dt_model1, commons.get_all_features(dt_model1, num_features1, cat_features1))

    dt_model2 = commons.train_model(commons.Models.DECISION_TREE, df_X2, df_y, num_features2, cat_features2)
    commons.interpret_model(dt_model2, commons.get_all_features(dt_model2, num_features2, cat_features2))
    """
    pass


def print_lime_local_results(df_X: pd.DataFrame, df_y: pd.Series, classifier: Pipeline, span: list, text_info: str):

    _, X_test, _, y_test = train_test_split(df_X,
                                            df_y,
                                            stratify=df_y,
                                            test_size=commons.TEST_SPLIT_SIZE,
                                            random_state=commons.RANDOM_NUMBER)

    _, cat_features = commons.divide_features(X_test)
    new_ohe_features = commons.get_ohe_cats(classifier, cat_features)

    # Transform the categorical feature's labels to a lime-readable format.
    categorical_names = {}
    for col in cat_features:
        categorical_names[X_test.columns.get_loc(col)] = [new_col.split("__")[1]
                                                          for new_col in new_ohe_features
                                                          if new_col.split("__")[0] == col]

    def custom_predict_proba(X, model):
        """
        Create a custom predict_proba for the model, so that it could be used in lime.
        :param X: Example to be classified.
        :param model: The model - classifier.
        :return: The probability that X will be classified as 1.
        """
        X_str = commons.convert_to_lime_format(X, categorical_names, col_names=X_test.columns, invert=True)
        return model.predict_proba(X_str)

    log.debug("Categorical names for lime: {}".format(categorical_names))

    explainer = LimeTabularExplainer(commons.convert_to_lime_format(X_test, categorical_names).values,
                                     mode="classification",
                                     feature_names=X_test.columns.tolist(),
                                     categorical_names=categorical_names,
                                     categorical_features=categorical_names.keys(),
                                     discretize_continuous=True,
                                     random_state=commons.RANDOM_NUMBER)

    for i in span:
        # Print print data of the person
        log.info("Person {}'s data: \n{}".format(i, X_test.iloc[i]))
        log.info("Person {}'s actual result: {}".format(i, df_y[i]))

        # Plot the results for that person
        custom_model_predict_proba = partial(custom_predict_proba, model=classifier)
        observation = commons.convert_to_lime_format(X_test.iloc[[i], :], categorical_names).values[0]
        explanation = explainer.explain_instance(observation,
                                                 custom_model_predict_proba,
                                                 num_features=len(num_features4))

        file = commons.EXAMPLES_DIR_LIME \
            + os.sep \
            + "person_" \
            + str(i) + "-"\
            + str(datetime.today().strftime('%Y%m%d-%H%M%S')) \
            + ".html"
        explanation.save_to_file(file)

        person_data_as_html = generate_html_string(i,
                                                   X_test.iloc[i],
                                                   df_y[i],
                                                   text_info)

        log.debug(person_data_as_html)

        with open(file, 'a') as html_file:
            html_file.write('\n')
            html_file.write(person_data_as_html)


def generate_html_string(id: int, data: pd.Series, result: int, text_info: str) -> str:
    """
    Generate html table containing the data of a person.
    :param id: The id of the person
    :param data: The data about the person (e.g. occupation, education, ...)
    :param result: Whether his salary is < 50k or >= 50k (0, 1).
    :param text_info: Text information that should be included in the explanation.
    :return: The html table containing information about the person
    """

    split_data = " ".join(str(data).split()).replace(" ", ",").split(",")
    html_start = """
    <table>
       <thead>
          <tr>
             <th>Person {}</th>
             <th />
          </tr>
       </thead>
       <tbody>""".format(id)
    html_base = ""
    for _ in range(len(data.values)):
        html_base = html_base + """\n          <tr>
             <td>{}</td>
             <td>{}</td>
          </tr>"""
    html_end = """\n          <tr>
             <td>salary</td>
             <td>{}</td>
          </tr>
          <tr>
             <td>excluded features: </td>
             <td>{}</td>
          </tr>
       </tbody>
    </table>
    """.format(result, text_info)
    html = html_start + html_base + html_end

    return html.format(*split_data)


def interpret_rf_models():
    """
    TODO: Implement function.
    :return:
    """

    """
    rf_model1 = commons.train_model(commons.Models.RANDOM_FOREST,
                                    commons.Splits.IMBALANCED,
                                    df_X1,
                                    df_y,
                                    num_features1,
                                    cat_features1)
    
    commons.interpret_model(rf_model1, commons.get_all_features(rf_model1, num_features1, cat_features1))
    """
    pass


def interpret_xgb_models():
    rand_numbers = []
    for i in range(commons.EXAMPLES_SPAN_LIME):
        rand_number = rd.randrange(0, int((len(df_X1.values)*commons.TEST_SPLIT_SIZE)))
        rand_numbers.append(rand_number)

    xgb_model1 = commons.train_model(commons.Models.XGB, commons.Splits.IMBALANCED, df_X1, df_y, num_features1,
                                     cat_features1)
    commons.interpret_model(xgb_model1, commons.get_all_features(xgb_model1, num_features1, cat_features1))
    print_lime_local_results(df_X1, df_y, xgb_model1, rand_numbers, "")

    log.info(
        "\n*******************************************************************************************************\n")

    xgb_model5 = commons.train_model(commons.Models.XGB, commons.Splits.IMBALANCED, df_X5, df_y, num_features5,
                                     cat_features5)
    commons.interpret_model(xgb_model5, commons.get_all_features(xgb_model5, num_features5, cat_features5))
    print_lime_local_results(df_X5, df_y, xgb_model5, rand_numbers, ", ".join(REMOVE_FEATURES_M2 +
                                                                              REMOVE_FEATURES_M4 +
                                                                              REMOVE_FEATURES_M5))

    log.info(
        "\n*******************************************************************************************************\n")

# Prepare the data
df = data.load_census()
df_X1 = df.drop(TARGET, axis=1)
# Model 2 and 3 use the same columns
df_X2 = df_X1.drop(REMOVE_FEATURES_M2, axis=1)
df_X4 = df_X2.drop(REMOVE_FEATURES_M4, axis=1)
df_X5 = df_X4.drop(REMOVE_FEATURES_M5, axis=1)
df_y = df[TARGET].map({" <=50K": 0, " >50K": 1})

log.debug("\n{}".format(df_y.value_counts()))

num_features1, cat_features1 = commons.divide_features(df_X1)
num_features2, cat_features2 = commons.divide_features(df_X2)
num_features4, cat_features4 = commons.divide_features(df_X4)
num_features5, cat_features5 = commons.divide_features(df_X5)

# interpret_lr_models()
# interpret_dt_models()
# interpret_rf_models()
interpret_xgb_models()
