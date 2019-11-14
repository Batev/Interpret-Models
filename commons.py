import pandas as pd

NUMERIC_TYPES = ["int", "float"]
RANDOM_NUMBER = 33


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
