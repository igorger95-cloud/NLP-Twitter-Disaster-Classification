# NLP Twitter Disaster Classification

Проект по классификации текстов с использованием классических методов NLP и современных Transformer-моделей.

## Цель проекта

Определить, описывает ли твит реальное чрезвычайное происшествие.

Целевая переменная:

- `0` — не связано с реальным происшествием
- `1` — связано с реальным происшествием

В проекте сравниваются классические методы обработки текста, Transformer embeddings и fine-tuning предобученной Transformer-модели.

---

# 1. Dataset

Используется датасет Kaggle:

**NLP Getting Started — Disaster Tweets**

Основные файлы:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

В репозитории исходные данные не хранятся.

Для получения данных необходимо скачать ZIP-архив с Kaggle и поместить CSV-файлы в папку:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv

2. Project Pipeline

Общий pipeline проекта:

Dataset
   ↓
EDA
   ↓
Text preprocessing
   ↓
Vectorization
   ↓
Classic ML
   ↓
Transformer embeddings
   ↓
Fine-tuning Transformer
   ↓
Evaluation
   ↓
Prediction
   ↓
submission.csv
3. Exploratory Data Analysis

EDA выполнен отдельно в Jupyter Notebook:

notebooks/EDA.ipynb

В рамках EDA были исследованы:

размеры train и test;
распределение целевой переменной;
пропущенные значения;
дубликаты;
длина текстов;
распределение классов;
особенности текстовых данных.
4. Text preprocessing

Для классических моделей были протестированы разные варианты обработки текста:

очистка текста;
удаление stop words;
stemming;
lemmatization;
различные варианты очистки.

Лучший вариант для классического подхода:

stemming

Для Transformer использовалась только минимальная обработка:

заполнение NaN;
преобразование текста в строковый тип;
удаление лишних пробелов.

Stemming, lemmatization и удаление stop words для Transformer не использовались.

Это позволяет сохранить естественный контекст текста.

5. Classic NLP

Были протестированы следующие подходы:

TF-IDF

TF-IDF преобразует текст в числовой вектор на основе важности слов в документах.

Использовалась:

TF-IDF + Logistic Regression

Результат:

F1 = 0.774938
Word2Vec

Word2Vec создаёт плотные векторные представления слов.

На его основе были протестированы:

Word2Vec + Logistic Regression
Word2Vec + Gradient Boosting
6. Transformer embeddings

Также был протестирован подход с использованием Sentence Transformer embeddings.

Полученные embeddings подавались в Logistic Regression.

Результат:

Sentence Transformer + Logistic Regression
F1 = 0.800635

Этот результат оказался выше классического TF-IDF подхода.

7. Fine-tuning Transformer

Для fine-tuning была выбрана модель:

distilbert-base-uncased
Почему DistilBERT

DistilBERT выбран потому что:

работает с английским языком;
является облегчённой версией BERT;
требует меньше памяти;
быстрее обучается;
хорошо подходит для коротких текстов;
подходит для обучения на Google Colab GPU T4.
8. Transformer preprocessing

В отличие от Classic NLP, текст перед Transformer не очищался агрессивно.

Не использовались:

stemming;
lemmatization;
удаление stop words;
удаление пунктуации;
TF-IDF.

Использовалась только минимальная обработка текста.

После этого текст передавался в tokenizer DistilBERT.

Параметры токенизации:

max_length = 128
truncation = True
9. Fine-tuning parameters

Использовались следующие параметры:

Model: distilbert-base-uncased

Learning rate: 2e-5
Batch size: 16
Epochs: 3
Weight decay: 0.01
Max length: 128
FP16: True

Обучение выполнялось на GPU Google Colab T4.

10. Evaluation

Основной метрикой проекта является:

F1-score

Дополнительно рассчитывались:

Accuracy;
Precision;
Recall.

| Подход                 | Векторизация         | Модель                         |     F1-score |
| ---------------------- | -------------------- | ------------------------------ | -----------: |
| Classic NLP            | TF-IDF               | Logistic Regression + Stemming |     0.774938 |
| Classic NLP            | Word2Vec             | Gradient Boosting              |     0.632997 |
| Classic NLP            | Word2Vec             | Logistic Regression            |     0.593070 |
| Transformer embeddings | Sentence Transformer | Logistic Regression            |     0.800635 |
| Fine-tuned Transformer | DistilBERT           | DistilBERT                     | **0.813990** |


12. Best model

Лучший результат показала модель:

Fine-tuned DistilBERT

F1-score:

0.813990

Она превзошла:

TF-IDF + Logistic Regression:
0.774938

Sentence Transformer + Logistic Regression:
0.800635

Таким образом, fine-tuning позволил получить лучший результат среди протестированных подходов.

13. Repository structure
NLP-Twitter-Disaster-Classification/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── classic_config.yaml
│   └── distilbert_config.yaml
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── EDA.ipynb
│
├── results/
│   └── metrics.csv
│
└── src/
    │
    ├── data/
    │   ├── load_data.py
    │   └── preprocessing.py
    │
    ├── features/
    │   ├── tfidf_features.py
    │   └── word2vec_features.py
    │
    ├── models/
    │   ├── train_classic.py
    │   ├── train_transformer.py
    │   └── predict.py
    │
    ├── evaluation/
    │   └── metrics.py
    │
    └── utils/
        └── config.py
14. Configuration

Параметры моделей вынесены в YAML-конфиги:

configs/classic_config.yaml
configs/distilbert_config.yaml

Это позволяет изменять параметры экспериментов без изменения основного Python-кода.

15. Installation

Клонировать репозиторий:

git clone https://github.com/USERNAME/NLP-Twitter-Disaster-Classification.git
cd NLP-Twitter-Disaster-Classification

Установить зависимости:

pip install -r requirements.txt
16. Running Classic NLP

После размещения данных в папке data/:

python src/models/train_classic.py

Результаты сохраняются в:

results/classic_metrics.csv
17. Running Transformer

Для fine-tuning DistilBERT:

python src/models/train_transformer.py

Обученная модель сохраняется в:

models/distilbert/
18. Creating predictions

После обучения модели:

python src/models/predict.py

Будет создан файл:

results/submission.csv

Формат:

id,target
0,1
1,0
2,1
...

Этот файл можно загрузить на Kaggle для получения итогового leaderboard score.

19. How to improve the results

Результат можно улучшить несколькими способами.

Для Classic NLP

Можно попробовать:

подобрать параметры TF-IDF;
изменить ngram_range;
увеличить max_features;
подобрать C для Logistic Regression;
использовать более продвинутые модели.
Для Transformer

Можно попробовать:

увеличить количество эпох;
подобрать learning rate;
изменить batch size;
подобрать max_length;
использовать другую предобученную модель;
провести hyperparameter search;
использовать class weights;
применить data augmentation.

Также можно протестировать более современные Transformer-модели, например RoBERTa или DeBERTa.

20. Conclusion

В проекте были сравнены классические методы NLP, Transformer embeddings и fine-tuning предобученной Transformer-модели.

Наилучший результат показал:

Fine-tuned DistilBERT
F1 = 0.813990

Таким образом, использование контекстных представлений и fine-tuning Transformer позволило получить лучший результат по сравнению с классическими методами NLP.
