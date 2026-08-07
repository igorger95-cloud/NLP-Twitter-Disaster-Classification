# NLP Twitter Disaster Classification

Проект по классификации текстов с использованием классических методов NLP и современных Transformer-моделей.

## Цель проекта

Определить, описывает ли твит реальное чрезвычайное происшествие.

Целевая переменная:

* `0` — твит не связан с реальным происшествием
* `1` — твит связан с реальным происшествием

В проекте сравниваются:

* классические методы NLP;
* Transformer embeddings;
* fine-tuning предобученной Transformer-модели.

---

# 1. Dataset

Используется датасет Kaggle:

**NLP Getting Started — Disaster Tweets**

Основные файлы:

* `train.csv`
* `test.csv`
* `sample_submission.csv`

Исходные CSV-файлы не являются частью исходного набора файлов проекта.

Для запуска необходимо разместить данные в папке:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

---

# 2. Project Pipeline

Общий pipeline проекта:

```text
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
```

---

# 3. Exploratory Data Analysis

EDA выполнен отдельно в Jupyter Notebook:

```text
notebooks/EDA.ipynb
```

В рамках EDA были исследованы:

* размеры train и test;
* распределение целевой переменной;
* пропущенные значения;
* дубликаты;
* длина текстов;
* распределение классов;
* особенности текстовых данных.

---

# 4. Text Preprocessing

Для классических моделей были протестированы разные варианты обработки текста:

* очистка текста;
* удаление stop words;
* stemming;
* lemmatization;
* различные варианты очистки.

В финальном классическом эксперименте использовался **stemming**.

Для Transformer использовалась только минимальная обработка:

* заполнение пропущенных значений;
* преобразование текста в строковый тип;
* удаление лишних пробелов.

Для Transformer не использовались:

* stemming;
* lemmatization;
* удаление stop words;
* удаление пунктуации.

Такой подход позволяет сохранить исходный контекст текста перед токенизацией.

---

# 5. Classic NLP

В классическом подходе были протестированы TF-IDF и Word2Vec.

## TF-IDF

TF-IDF преобразует текст в числовой вектор на основе важности слов и n-грамм в документах.

Использовалась комбинация:

**TF-IDF + Logistic Regression + Stemming**

Результат:

**F1 = 0.781609**

## Word2Vec

Word2Vec создаёт плотные векторные представления слов.

Для каждого текста использовалось усреднение векторов его слов.

На основе Word2Vec были протестированы:

**Word2Vec + Logistic Regression**

F1 = **0.644497**

**Word2Vec + Gradient Boosting**

F1 = **0.683706**

Таким образом, среди протестированных классических методов лучший результат показал:

**TF-IDF + Logistic Regression + Stemming**

F1 = **0.781609**

---

# 6. Transformer Embeddings

Также был протестирован подход с использованием Sentence Transformer embeddings.

Полученные embeddings подавались в Logistic Regression.

Результат:

**Sentence Transformer + Logistic Regression**

F1 = **0.800635**

Этот результат оказался выше результатов протестированных классических моделей.

---

# 7. Fine-tuning Transformer

Для fine-tuning была выбрана модель:

**distilbert-base-uncased**

## Почему DistilBERT

DistilBERT выбран потому что:

* работает с английским языком;
* является облегчённой версией BERT;
* требует меньше вычислительных ресурсов;
* быстрее обучается;
* подходит для коротких текстов;
* позволяет выполнять обучение в Google Colab на GPU T4.

---

# 8. Transformer Preprocessing

В отличие от Classic NLP, текст перед Transformer не очищался агрессивно.

Не использовались:

* stemming;
* lemmatization;
* удаление stop words;
* удаление пунктуации;
* TF-IDF.

Использовалась только минимальная обработка текста.

После этого текст передавался в tokenizer DistilBERT.

Параметры токенизации:

```text
max_length = 128
truncation = True
```

---

# 9. Fine-tuning Parameters

Использовались следующие параметры:

| Parameter     |                     Value |
| ------------- | ------------------------: |
| Model         | `distilbert-base-uncased` |
| Learning rate |                    `2e-5` |
| Batch size    |                      `16` |
| Epochs        |                       `3` |
| Weight decay  |                    `0.01` |
| Max length    |                     `128` |
| FP16          |                    `True` |

Обучение выполнялось на GPU Google Colab T4.

---

# 10. Evaluation

Основной метрикой проекта является:

**F1-score**

Дополнительно рассчитывались:

* Accuracy;
* Precision;
* Recall.

---

# 11. Results

Итоговые результаты экспериментов:

| Approach               | Vectorization        | Model                          |     F1-score |
| ---------------------- | -------------------- | ------------------------------ | -----------: |
| Classic NLP            | TF-IDF               | Logistic Regression + Stemming |     0.781609 |
| Classic NLP            | Word2Vec             | Logistic Regression            |     0.644497 |
| Classic NLP            | Word2Vec             | Gradient Boosting              |     0.683706 |
| Transformer embeddings | Sentence Transformer | Logistic Regression            |     0.800635 |
| Fine-tuned Transformer | DistilBERT           | Fine-tuned DistilBERT          | **0.813990** |

Результаты сохранены в:

```text
results/metrics.csv
```

---

# 12. Best Model

Лучший результат показала модель:

**Fine-tuned DistilBERT**

F1-score:

**0.813990**

Сравнение с другими подходами:

* TF-IDF + Logistic Regression — **0.781609**
* Word2Vec + Logistic Regression — **0.644497**
* Word2Vec + Gradient Boosting — **0.683706**
* Sentence Transformer + Logistic Regression — **0.800635**
* Fine-tuned DistilBERT — **0.813990**

Таким образом, fine-tuning DistilBERT позволил получить лучший результат среди протестированных подходов.

---

# 13. Repository Structure

```text
NLP-Twitter-Disaster-Classification/
│
├── README.md
├── requirements.txt
├── .gitignore
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
│   ├── metrics.csv
│   └── submission.csv
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
```

Обученная Transformer-модель создаётся в папке:

```text
models/distilbert/
```

Эта директория появляется после запуска `train_transformer.py` и не требуется для хранения исходного кода проекта.

---

# 14. Configuration

Параметры экспериментов вынесены в YAML-конфиги:

```text
configs/classic_config.yaml
configs/distilbert_config.yaml
```

### Classic NLP

`classic_config.yaml` содержит параметры:

* TF-IDF;
* Logistic Regression;
* Gradient Boosting;
* Word2Vec.

### Transformer

`distilbert_config.yaml` содержит параметры:

* модели;
* токенизации;
* обучения;
* train/validation split.

Это позволяет изменять параметры экспериментов без изменения основного Python-кода.

---

# 15. Installation

Клонировать репозиторий:

```bash
git clone https://github.com/USERNAME/NLP-Twitter-Disaster-Classification.git
cd NLP-Twitter-Disaster-Classification
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

Подготовить данные в папке `data/`:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

---

# 16. Running Classic NLP

После размещения данных в папке `data/`:

```bash
python src/models/train_classic.py
```

Скрипт выполняет:

```text
Preprocessing
    ↓
Train/validation split
    ↓
TF-IDF + Logistic Regression
    ↓
Word2Vec
    ├── Logistic Regression
    └── Gradient Boosting
```

Результаты сохраняются в:

```text
results/metrics.csv
```

---

# 17. Running Transformer

Для fine-tuning DistilBERT:

```bash
python src/models/train_transformer.py
```

Обученная модель сохраняется в:

```text
models/distilbert/
```

---

# 18. Creating Predictions

После обучения DistilBERT:

```bash
python src/models/predict.py
```

Будет создан файл:

```text
results/submission.csv
```

Формат:

```text
id,target
0,1
1,0
2,1
...
```

Файл можно загрузить на Kaggle для получения итогового leaderboard score.

---

# 19. How to Improve the Results

## Для Classic NLP

Можно попробовать:

* подобрать параметры TF-IDF;
* изменить `ngram_range`;
* увеличить `max_features`;
* подобрать `C` для Logistic Regression;
* изменить параметры Word2Vec;
* использовать более продвинутые модели.

## Для Transformer

Можно попробовать:

* увеличить количество эпох;
* подобрать learning rate;
* изменить batch size;
* подобрать `max_length`;
* использовать другую предобученную модель;
* провести hyperparameter search;
* использовать class weights;
* применить data augmentation.

Также можно протестировать другие Transformer-модели, например RoBERTa или DeBERTa.

---

# 20. Conclusion

В проекте были сравнены классические методы NLP, Transformer embeddings и fine-tuning предобученной Transformer-модели.

Результаты показали постепенное улучшение качества:

```text
Word2Vec + Logistic Regression       0.644497
Word2Vec + Gradient Boosting         0.683706
TF-IDF + Logistic Regression         0.781609
Sentence Transformer + LR            0.800635
Fine-tuned DistilBERT                0.813990
```

Наилучший результат показал:

**Fine-tuned DistilBERT**

**F1 = 0.813990**

Таким образом, использование контекстных представлений и fine-tuning Transformer позволило получить лучший результат среди протестированных методов.


