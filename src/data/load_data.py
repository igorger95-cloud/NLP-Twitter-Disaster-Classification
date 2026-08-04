import pandas as pd


def load_data(train_path, test_path):
    """
    Load train and test datasets.

    Parameters
    ----------
    train_path : str
        Path to train.csv.

    test_path : str
        Path to test.csv.

    Returns
    -------
    train : pd.DataFrame
        Training dataset.

    test : pd.DataFrame
        Test dataset.
    """

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test
