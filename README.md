# NLP Twitter Disaster Classification

Проект по классификации текстов с использованием классических методов NLP и современных Transformer-моделей.

## Цель проекта

Определить, описывает ли твит реальное чрезвычайное происшествие.

Целевая переменная:

* `0` — не связано с реальным происшествием
* `1` — связано с реальным происшествием

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

Исходные CSV-файлы не хранятся в GitHub-репозитории.

Для запуска проекта необходимо скачать датасет и разместить файлы в папке:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Подробнее о данных:

```text
data/README.md
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

EDA отделён от основного pipeline проекта и используется для анализа данных.

---

# 4. Text Preprocessing

Для классических моделей были протестированы разные варианты обработки текста:

* очистка текста;
* удаление stop words;
* stemming;
* lemmatization;
* различные варианты очистки.

Для итогового классического эксперимента использовался:

**stemming**

Классическая обработка включает:

* приведение текста к нижнему регистру;
* удаление URL;
* удаление упоминаний пользователей;
* удаление лишних символов;
* удаление лишних пробелов;
* stemming.

Для Transformer использовалась только минимальная обработка:

* заполнение пропущенных значений;
* преобразование текста в строковый тип;
* удаление лишних пробелов.

Stemming, lemmatization и удаление stop words для Transformer не использовались.

Это позволяет сохранить исходный контекст текста перед токенизацией.

---

# 5. Classic NLP

Для классического NLP были протестированы TF-IDF и Word2Vec.

## TF-IDF

TF-IDF преобразует текст в числовое представление на основе важности слов и n-грамм в документах.

Использовалась комбинация:

**TF-IDF + Logistic Regression + Stemming**

Результат:

**F1 = 0.781609**

Параметры TF-IDF:

```text
max_features = 5000
ngram_range = (1, 2)
```

Параметры Logistic Regression:

```text
C = 1.0
max_iter = 1000
```

---

## Word2Vec

Word2Vec создаёт плотные векторные представления слов.

Для получения представления документа использовалось усреднение векторов слов.

Были протестированы две модели:

### Word2Vec + Logistic Regression

**F1 = 0.645008**

### Word2Vec + Gradient Boosting

**F1 = 0.659677**

Параметры Word2Vec вынесены в:

```text
configs/classic_config.yaml
```

---

# 6. Transformer Embeddings

Также был протестирован подход с использованием Sentence Transformer embeddings.

Полученные embeddings подавались в Logistic Regression.

Результат:

**Sentence Transformer + Logistic Regression**

**F1 = 0.800635**

Этот результат оказался выше всех протестированных классических подходов.

---

# 7. Fine-tuning Transformer

Для fine-tuning была выбрана модель:

**distilbert-base-uncased**

## Почему DistilBERT

DistilBERT выбран потому что:

* работает с английским языком;
* является облегчённой версией BERT;
* требует меньше памяти;
* быстрее обучается;
* хорошо подходит для коротких текстов;
* подходит для обучения на Google Colab GPU T4.

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

Конфигурация Transformer хранится в:

```text
configs/distilbert_config.yaml
```

---

# 10. Evaluation

Основной метрикой проекта является:

**F1-score**

Дополнительно рассчитывались:

* Accuracy;
* Precision;
* Recall.

Для классических моделей метрики рассчитываются с помощью общей функции:

```text
src/evaluation/metrics.py
```

---

# 11. Results

Итоговое сравнение всех протестированных подходов:

| Approach               | Vectorization        | Model                          |     F1-score |
| ---------------------- | -------------------- | ------------------------------ | -----------: |
| Classic NLP            | TF-IDF               | Logistic Regression + Stemming |     0.781609 |
| Classic NLP            | Word2Vec             | Logistic Regression            |     0.645008 |
| Classic NLP            | Word2Vec             | Gradient Boosting              |     0.659677 |
| Transformer embeddings | Sentence Transformer | Logistic Regression            |     0.800635 |
| Fine-tuned Transformer | DistilBERT           | Fine-tuned DistilBERT          | **0.813990** |

Результаты экспериментов сохранены в:

```text
results/metrics.csv
```

---

# 12. Best Model

Лучший результат показала модель:

**Fine-tuned DistilBERT**

F1-score:

**0.813990**

Сравнение с основными конкурентами:

| Model                                      |     F1-score |
| ------------------------------------------ | -----------: |
| TF-IDF + Logistic Regression               |     0.781609 |
| Sentence Transformer + Logistic Regression |     0.800635 |
| Fine-tuned DistilBERT                      | **0.813990** |

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

---

# 14. Configuration

Параметры моделей вынесены в YAML-конфиги:

```text
configs/classic_config.yaml
configs/distilbert_config.yaml
```

Это позволяет изменять параметры экспериментов без изменения основного Python-кода.

Пример параметров классического NLP:

```yaml
tfidf:
  max_features: 5000
  ngram_range:
    - 1
    - 2

logistic_regression:
  C: 1.0
  max_iter: 1000

gradient_boosting:
  n_estimators: 100

word2vec:
  vector_size: 100
  window: 5
  min_count: 1
  workers: 4
  epochs: 10
```

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

Основные библиотеки проекта:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
gensim
torch
transformers
datasets
accelerate
evaluate
pyyaml
```

---

# 16. Running Classic NLP

После размещения данных в папке `data/`:

```bash
python src/models/train_classic.py
```

Скрипт запускает:

1. TF-IDF + Logistic Regression;
2. Word2Vec + Logistic Regression;
3. Word2Vec + Gradient Boosting.

Результаты классических экспериментов сохраняются в:

```text
results/classic_metrics.csv
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

После обучения модели:

```bash
python src/models/predict.py
```

Будет создан файл:

```text
results/submission.csv
```

Формат файла:

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

Результат можно улучшить несколькими способами.

## Для Classic NLP

Можно попробовать:

* подобрать параметры TF-IDF;
* изменить `ngram_range`;
* увеличить `max_features`;
* подобрать `C` для Logistic Regression;
* изменить параметры Word2Vec;
* использовать другие модели классификации.

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

Также можно протестировать более современные Transformer-модели, например RoBERTa или DeBERTa.

---

# 20. Conclusion

В проекте были сравнены три подхода к классификации текстов:

1. Classic NLP;
2. Transformer embeddings;
3. Fine-tuning Transformer.

Полученные результаты показывают постепенное улучшение качества:

```text
Classic NLP
     ↓
TF-IDF + Logistic Regression
F1 = 0.781609

     ↓
Transformer embeddings
F1 = 0.800635

     ↓
Fine-tuned DistilBERT
F1 = 0.813990
```

Наилучший результат показал:

**Fine-tuned DistilBERT — F1 = 0.813990**

Таким образом, использование контекстных представлений и fine-tuning предобученной Transformer-модели позволило получить лучший результат среди протестированных подходов.


