# NLP Twitter Disaster Classification

Проект по классификации текстов с использованием классических методов NLP и современных Transformer-моделей.

## Цель проекта

Определить, описывает ли твит реальное чрезвычайное происшествие.

Целевая переменная:

* `0` — твит не связан с реальным происшествием
* `1` — твит связан с реальным происшествием

В проекте сравниваются три подхода:

1. Classic NLP + классические ML-модели
2. Предобученные Transformer embeddings + классическая ML-модель
3. Fine-tuning предобученной Transformer-модели

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
* баланс классов;
* пропущенные значения;
* дубликаты;
* длина текстов;
* распределение длины текстов;
* особенности текстовых данных.

EDA отделён от основного pipeline проекта и используется только для анализа данных.

---

# 4. Text Preprocessing

## Classic NLP

Для классических моделей были протестированы разные варианты обработки текста:

* приведение к нижнему регистру;
* очистка текста;
* удаление URL;
* удаление упоминаний пользователей;
* удаление лишних символов;
* удаление лишних пробелов;
* удаление stop words;
* stemming;
* lemmatization;
* варианты без дополнительной морфологической обработки.

Для итогового классического эксперимента использовался **stemming**.

Основной preprocessing:

```text
lowercase
    ↓
remove URLs
    ↓
remove mentions
    ↓
remove unnecessary symbols
    ↓
remove extra spaces
    ↓
stemming
```

## Transformer models

Для Transformer-моделей использовалась минимальная предобработка:

* заполнение пропущенных значений;
* преобразование текста в строковый тип;
* удаление лишних пробелов.

Не использовались:

* stemming;
* lemmatization;
* удаление stop words;
* агрессивное удаление пунктуации.

Это позволяет сохранить исходный контекст текста перед токенизацией.

---

# 5. Classic NLP

Для классического NLP были протестированы:

* TF-IDF;
* Word2Vec.

В качестве классификаторов использовались:

* Logistic Regression;
* Gradient Boosting.

## 5.1 TF-IDF

TF-IDF преобразует текст в числовое представление на основе важности слов и n-грамм в документах.

Итоговая комбинация:

```text
TF-IDF
   +
Logistic Regression
   +
Stemming
```

Результат:

**F1 = 0.781609**

Основные параметры:

```text
max_features = 5000
ngram_range = (1, 2)
```

Logistic Regression:

```text
C = 1.0
max_iter = 1000
```

---

## 5.2 Word2Vec

Word2Vec создаёт плотные векторные представления слов.

Для получения представления документа использовалось усреднение векторов слов.

Были протестированы две комбинации.

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

Для получения контекстных представлений текста использовалась модель:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Модель преобразует предложение или текст в плотный вектор размерности 384. Она предназначена для получения sentence embeddings и используется для задач семантического поиска, similarity и других задач обработки текста.

## Почему all-MiniLM-L6-v2

При выборе модели учитывались:

* результаты benchmark MTEB;
* качество sentence embeddings;
* небольшой размер модели;
* скорость получения embeddings;
* требования к памяти;
* возможность эффективного запуска в Google Colab.

Модель присутствует в экосистеме Sentence Transformers и имеет результаты на MTEB. Например, в карточке модели представлены результаты оценки MTEB.

Таким образом, `all-MiniLM-L6-v2` была выбрана как хороший компромисс между качеством и скоростью для задачи классификации коротких английских текстов.

## Classification

Полученные embeddings подавались в:

```text
Logistic Regression
```

Результат:

**F1 = 0.800635**

Этот результат оказался выше результатов протестированных классических подходов.

Скрипт эксперимента:

```text
src/models/train_sentence_transformer.py
```

---

# 7. Fine-tuning Transformer

Для fine-tuning была выбрана модель:

```text
distilbert-base-uncased
```

## Почему DistilBERT

DistilBERT выбран потому что:

* работает с английским языком;
* является облегчённой версией BERT;
* требует меньше памяти;
* быстрее обучается;
* хорошо подходит для коротких текстов;
* позволяет выполнять fine-tuning в Google Colab на GPU T4.

Для обучения использовалась библиотека:

```text
transformers
```

## Preprocessing

Для DistilBERT использовалась минимальная обработка текста.

Не применялись:

* stemming;
* lemmatization;
* удаление stop words;
* агрессивное удаление пунктуации;
* TF-IDF.

После минимальной очистки текст передавался в tokenizer DistilBERT.

Параметры токенизации:

```text
max_length = 128
truncation = True
```

---

# 8. Fine-tuning Parameters

Основные параметры обучения:

| Parameter     | Value                     |
| ------------- | ------------------------- |
| Model         | `distilbert-base-uncased` |
| Learning rate | `2e-5`                    |
| Batch size    | `16`                      |
| Epochs        | `3`                       |
| Weight decay  | `0.01`                    |
| Max length    | `128`                     |
| FP16          | `True`                    |

Обучение выполнялось на GPU Google Colab T4.

Конфигурация Transformer хранится в:

```text
configs/distilbert_config.yaml
```

---

# 9. Evaluation

Основной метрикой проекта является:

**F1-score**

Дополнительно рассчитывались:

* Accuracy;
* Precision;
* Recall.

Метрики рассчитываются с помощью общей функции:

```text
src/evaluation/metrics.py
```

---

# 10. Results

Итоговое сравнение всех протестированных подходов:

| Approach               | Vectorization        | Model                          |     F1-score |
| ---------------------- | -------------------- | ------------------------------ | -----------: |
| Classic NLP            | TF-IDF               | Logistic Regression + Stemming | **0.781609** |
| Classic NLP            | Word2Vec             | Logistic Regression            |     0.645008 |
| Classic NLP            | Word2Vec             | Gradient Boosting              |     0.659677 |
| Transformer embeddings | Sentence Transformer | Logistic Regression            | **0.800635** |
| Fine-tuned Transformer | DistilBERT           | Fine-tuned DistilBERT          | **0.813990** |

Результаты экспериментов сохранены в:

```text
results/metrics.csv
```

---

# 11. Best Model

Лучший результат показала:

```text
Fine-tuned DistilBERT
```

F1-score:

```text
0.813990
```

Сравнение лучших подходов:

| Model                                      |     F1-score |
| ------------------------------------------ | -----------: |
| TF-IDF + Logistic Regression               |     0.781609 |
| Sentence Transformer + Logistic Regression |     0.800635 |
| Fine-tuned DistilBERT                      | **0.813990** |

Fine-tuning DistilBERT позволил получить лучший результат среди протестированных подходов.

---

# 12. Quality / Speed Comparison

По качеству лучшей является:

**Fine-tuned DistilBERT — F1 = 0.813990**

Sentence Transformer занимает второе место:

**all-MiniLM-L6-v2 + Logistic Regression — F1 = 0.800635**

Классический подход:

**TF-IDF + Logistic Regression — F1 = 0.781609**

По скорости и требованиям к ресурсам классические модели являются наиболее простыми и быстрыми.

Sentence Transformer представляет компромисс между качеством и скоростью получения embeddings.

Fine-tuned DistilBERT обеспечивает лучшее качество, но требует больше вычислительных ресурсов и времени на обучение.

Таким образом:

| Approach                     | Quality    | Speed / Resources                |
| ---------------------------- | ---------- | -------------------------------- |
| TF-IDF + Logistic Regression | Среднее    | **Очень быстро / мало ресурсов** |
| Sentence Transformer + LR    | Высокое    | Быстро после загрузки модели     |
| Fine-tuned DistilBERT        | **Лучшее** | Медленнее / требует GPU          |

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
    │   ├── train_sentence_transformer.py
    │   ├── train_transformer.py
    │   └── predict.py
    │
    ├── evaluation/
    │   └── metrics.py
    │
    └── utils/
        └── config.py
```

EDA находится отдельно в notebook, остальные основные этапы проекта реализованы в Python-скриптах.

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
git clone https://github.com/igorger95-cloud/NLP-Twitter-Disaster-Classification.git
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
sentence-transformers
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

Результаты сохраняются в:

```text
results/metrics.csv
```

---

# 17. Running Sentence Transformer

Для получения embeddings с помощью Sentence Transformer:

```bash
python src/models/train_sentence_transformer.py
```

Используется:

```text
all-MiniLM-L6-v2
```

После получения embeddings выполняется классификация с помощью Logistic Regression.

Результат эксперимента:

```text
F1 = 0.800635
```

---

# 18. Running Fine-tuned Transformer

Для fine-tuning DistilBERT:

```bash
python src/models/train_transformer.py
```

Обученная модель сохраняется в директории:

```text
models/distilbert/
```

---

# 19. Creating Predictions

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

Файл можно использовать для отправки предсказаний на Kaggle.

---

# 20. How to Improve the Results

## Classic NLP

Возможные улучшения:

* подобрать параметры TF-IDF;
* изменить `ngram_range`;
* увеличить `max_features`;
* подобрать `C` для Logistic Regression;
* изменить параметры Word2Vec;
* использовать другие модели классификации;
* попробовать ансамбли моделей.

## Transformer

Возможные улучшения:

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

# 21. Conclusion

В проекте были сравнены три основных подхода:

1. Classic NLP;
2. Transformer embeddings + Classic ML;
3. Fine-tuning Transformer.

Получены следующие результаты:

```text
TF-IDF + Logistic Regression
F1 = 0.781609

        ↓

Sentence Transformer + Logistic Regression
F1 = 0.800635

        ↓

Fine-tuned DistilBERT
F1 = 0.813990
```

Лучший результат показал:

**Fine-tuned DistilBERT — F1 = 0.813990**

Classic NLP оказался самым быстрым и простым по вычислительным требованиям.

Sentence Transformer позволил получить более высокое качество без fine-tuning большой языковой модели и представляет хороший компромисс между качеством и скоростью.

Fine-tuned DistilBERT показал максимальное качество среди протестированных моделей, но требует больше вычислительных ресурсов и времени на обучение.

Таким образом, эксперимент показывает постепенное улучшение качества при переходе от классических методов NLP к контекстным embeddings и последующему fine-tuning Transformer-модели.

