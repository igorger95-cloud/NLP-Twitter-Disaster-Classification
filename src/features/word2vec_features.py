import numpy as np
from gensim.models import Word2Vec


def train_word2vec(
    texts,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=10
):
    """
    Train Word2Vec model on tokenized texts.
    """

    tokenized_texts = [
        text.split()
        for text in texts
    ]

    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )

    return model


def texts_to_vectors(texts, word2vec_model):
    """
    Convert texts into averaged Word2Vec vectors.
    """

    vectors = []

    vector_size = word2vec_model.vector_size

    for text in texts:
        words = text.split()

        word_vectors = [
            word2vec_model.wv[word]
            for word in words
            if word in word2vec_model.wv
        ]

        if word_vectors:
            vector = np.mean(word_vectors, axis=0)
        else:
            vector = np.zeros(vector_size)

        vectors.append(vector)

    return np.array(vectors)
