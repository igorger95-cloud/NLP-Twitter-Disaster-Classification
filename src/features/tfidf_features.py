from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_features(
    train_texts,
    test_texts,
    max_features=5000,
    ngram_range=(1, 2)
):
    """
    Create TF-IDF features for train and test texts.

    Parameters
    ----------
    train_texts : array-like
        Training texts.

    test_texts : array-like
        Test texts.

    max_features : int
        Maximum vocabulary size.

    ngram_range : tuple
        Range of n-grams.

    Returns
    -------
    X_train : sparse matrix
        TF-IDF features for training data.

    X_test : sparse matrix
        TF-IDF features for test data.

    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer
