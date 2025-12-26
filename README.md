# Fake News Detection

Проект по курсу MLOps для задачи классификации фейковых новостей с
использованием DistilBERT. Основная цель проекта — создание production-ready
пайплайна машинного обучения, включающего управление данными, эксперименты,
тестирование и развертывание модели.

## Аннотация

### Постановка задачи

Фейковые новости (fake news) — это ложная или вводящая в заблуждение информация,
распространяемая под видом новостей. Распространение такой информации стало
серьезной проблемой для современного общества, влияя на общественное мнение,
политические процессы и доверие к медиа.

Цель данного проекта — разработать автоматизированную систему на основе
NLP-модели (DistilBERT), способную классифицировать новостные тексты на
**достоверные (Real)** и **недостоверные (Fake)**.

### Зачем это нужно?

1.  **Образовательные цели**: Проект демонстрирует полный цикл MLOps: от
    обработки данных и обучения трансформеров до версионирования данных (DVC),
    конфигурации (Hydra), логирования (MLflow) и деплоя (Triton Inference
    Server).
2.  **Практическая значимость**: Автоматическая фильтрация контента может помочь
    модераторам платформ и пользователям быстрее выявлять потенциально ложную
    информацию.

## Формат входных и выходных данных

Датасет состоит из двух CSV-файлов: `Fake.csv` и `True.csv`. Каждая запись
содержит заголовок (`title`), текст статьи (`text`), тему (`subject`) и дату
(`date`).

- **Входные данные модели**: Текст (объединение заголовка и основного текста
  статьи).
- **Препроцессинг**: Токенизация с использованием `DistilBertTokenizer`.
  Максимальная длина последовательности (max_length) ограничена 512 токенами.
- **Выходные данные**: Вероятность принадлежности к классу Fake (0 - Real, 1 -
  Fake).

Данные сбалансированы (примерно поровну Real и Fake новостей), что упрощает
обучение.

## Метрики

Для оценки качества модели используются следующие метрики:

- **F1-score**: Гармоническое среднее между точностью (Precision) и полнотой
  (Recall). Это основная метрика, так как она дает сбалансированную оценку
  качества классификации.
- **Accuracy**: Общая доля правильных ответов.
- **ROC-AUC**: Площадь под кривой ошибок, позволяющая оценить качество
  ранжирования вероятностей моделью.

## Источник данных

Используется открытый датасет с Kaggle:
[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## Моделирование

### Подготовка данных

- Объединение `title` и `text` в единый входной текст.
- Разметка: 1 для Fake, 0 для Real.
- Разбиение на train/val/test (по умолчанию 70/15/15).
- Токенизация с помощью предобученного токенизатора `distilbert-base-uncased`.

### Архитектура нейронной сети

Модель `FakeNewsModel` основана на предобученной модели **DistilBERT**
(`distilbert-base-uncased`).

Архитектура:

1.  **Backbone**: `DistilBertModel` (трансформер, ~66M параметров).
2.  **Pooling**: Используется вектор CLS-токена (первого токена
    последовательности).
3.  **Classifier Head**:
    - Dropout (0.2) для регуляризации.
    - Linear слой (`hidden_size` -> 1) для бинарной классификации.
4.  **Loss**: `BCEWithLogitsLoss`.

## Структура проекта

```
fake-news-detector/
├── .dvc/                  # DVC configs
├── conf/                  # Hydra configs
├── data/                  # Data (managed by DVC)
├── fake_news_detector/    # Main package
│   ├── commands.py        # CLI commands
│   ├── dataset.py         # Data preprocessing
│   ├── model.py           # LightningModule
│   ├── train.py           # Training script
│   └── infer.py           # Inference script
├── tests/                 # Unit tests (TODO)
├── triton/                # Triton Inference Server configs
├── .pre-commit-config.yaml # Git hooks
├── pyproject.toml         # Dependencies (uv/pep621)
└── README.md              # Project documentation
```

## Setup

Для начала работы с проектом необходимо настроить окружение. В качестве
менеджера пакетов используется `uv`.

1.  **Установите uv** (если не установлен):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Клонируйте репозиторий и перейдите в него**:

    ```bash
    git clone git@github.com:sergeyr0man0v/fake-news-detector.git
    cd fake-news-detector
    ```

3.  **Создайте виртуальное окружение и установите зависимости**:

    ```bash
    uv sync
    ```

    Это создаст виртуальное окружение `.venv` и установит все необходимые
    библиотеки из `uv.lock`.

4.  **Активируйте виртуальное окружение**:

    ```bash
    source .venv/bin/activate
    ```

5.  **Установите pre-commit хуки**: Это обеспечит проверку качества кода (black,
    isort, flake8) перед каждым коммитом.

    ```bash
    pre-commit install
    ```

6.  **Инициализируйте DVC** (если еще не сделано):
    ```bash
    dvc init
    ```

## Data Management

Данные управляются с помощью DVC.

Для **загрузки данных** используйте встроенную команду (требует Kaggle API
credentials или можно настроить загрузку из DVC remote, если он
сконфигурирован):

```bash
python fake_news_detector/commands.py download_data
```

Команда скачает датасет "Fake and Real News Dataset" и поместит его в
`data/raw`.

## Train

Для запуска тренировки используется `hydra` для управления конфигурацией.
Конфиги находятся в папке `conf/`.

**Запуск тренировки с параметрами по умолчанию:**

```bash
python fake_news_detector/train.py
```

**Переопределение гиперпараметров через CLI:**

Вы можете менять параметры на лету, не изменяя файлы конфигов:

```bash
# Изменение количества эпох и размера батча
python fake_news_detector/train.py training.max_epochs=10 model.batch_size=16

# Изменение путей к данным
python fake_news_detector/train.py data.data_dir=data/custom_data
```

После тренировки метрики и артефакты логируются в MLflow (по умолчанию локально
в `./mlruns`). Графики метрик также автоматически генерируются в папку `plots/`
после завершения.

## Production Preparation

Подготовка модели к продакшену включает конвертацию весов в оптимизированные
форматы.

**Экспорт в ONNX:** Позволяет запускать модель на различных платформах с помощью
ONNX Runtime.

```bash
python fake_news_detector/commands.py export_onnx --model_path=models/best_model.ckpt
```

Результат: `model.onnx`

**Экспорт в TensorRT:** Требует наличия NVIDIA GPU и установленных драйверов.
Обеспечивает максимальную производительность на GPU.

```bash
python fake_news_detector/commands.py export_tensorrt --onnx_path=models/model.onnx
```

Результат: `model.plan`

## Infer

Запуск инференса на новых данных.

**CLI Inference:** Скрипт принимает путь к чекпоинту модели и текст (или файл с
текстом).

````bash
# Инференс на строке текста
python fake_news_detector/infer.py +model_path="'models/best_model.ckpt'" +text="Scientists discovered a new planet made of cheese."

Формат вывода в консоль:
```text
Text: Scientists discovered a new planet made of cheese...
Prediction: FAKE
Probability (Fake): 0.9854
Confidence: 0.9854
````

## Inference Server (Triton)

Для развертывания модели используется Triton Inference Server с веб-интерфейсом.

**Структура сервиса:**

```
triton
├── docker-compose.yml
├── Dockerfile          # Образ для веб-интерфейса
└── sources
    ├── model.onnx      # Сюда нужно поместить экспортированную модель
    ├── triton.py       # Логика клиента Triton
    └── static          # Веб-интерфейс
```

**Запуск через Docker Compose:**

1.  Убедитесь, что модель экспортирована в ONNX (`model.onnx` должен находиться
    в `models/`).
2.  Скопируйте `model.onnx` в папку `triton/sources/`:
    ```bash
    cp models/model.onnx triton/sources/model.onnx
    cp models/model.onnx.data triton/sources/model.onnx.data
    ```
3.  Перейдите в директорию `triton`:
    ```bash
    cd triton
    ```
4.  Запустите сервисы:
    ```bash
    docker-compose up --build
    ```
5.  Откройте браузер по адресу [http://127.0.0.1:8080](http://127.0.0.1:8080).

**Остановка сервисов:**

```bash
docker-compose down
```
