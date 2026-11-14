# Two-Stage Information Retrieval & Neural Reranker (MIRACL-RU)

Двухэтапная система семантического поиска и переранжирования ответов (**BM25 + CrossEncoder**) для голосового помощника на базе русскоязычного корпуса **MIRACL** (~9.5 млн документов). Проект выполнен в рамках учебной практики в **VK Education**.

---

## Архитектура решения

Для баланса между скоростью поиска по многомиллионному корпусу и высокой точностью ранжирования реализован стандартный индустриальный пайплайн **Retrieve & Re-rank**:
1. **Stage 1 — Кандидатный отбор (Retrieval):** быстрый лексический поиск по корпусу с помощью алгоритма **BM25** (`rank_bm25`) отбирает топ-50 наиболее похожих документов-кандидатов.
2. **Stage 2 — Нейросетевое переранжирование (Reranking):** дообученный трансформер **CrossEncoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) оценивает релевантность каждой пары `(query, document)` с помощью механизма кросс-внимания и выстраивает итоговый топ-K ответов.

---

## Результаты на валидации (`dev`, 1252 запроса)

Обучение проводилось на сбалансированной выборке из **22 049 пар** (~8 000 позитивных примеров из `qrels` + 14 049 сложных и случайных негативов; `AdamW`, `lr=2e-5`, `warmup=10%`, `max_length=256`).

| Метрика | Значение | Интерпретация |
|---|---:|---|
| **Top-10 Hit** | **0.9944** | В 99.44% запросов релевантный документ попадает в топ-10 выдачи |
| **MRR@10** | **0.6657** | Средний обратный ранг первого правильного ответа (в среднем 1–2 позиция) |
| **P@1** | **0.5040** | В 50.4% случаев самый первый ответ модели (`Rank 1`) является точным |

### Пример работы инференса

Запрос: *"Кто такой Юрий Гагарин?"* (`--top-k 3`):

```text
=== Rank 1 | score=4.4113 | docid=2397362#1 ===
Космический полёт
Первый человек, полетевший в космос — советский космонавт Юрий Гагарин. 12 апреля 1961 года на космическом корабле «Восток», запущенном с космодрома «Байконур», он совершил полет по орбите вокруг Земли, 108 минут находясь в космосе, и успешно вернулся на родную планету.
```

---

## Структура проекта

```text
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/                   # Исходные corpus/*.jsonl, topics/*.tsv, qrels/*.tsv
│   └── processed/             # Отфильтрованный corpus.jsonl, train_pairs.jsonl, dev_pairs.jsonl
├── models/
│   └── reranker/              # Веса обученного CrossEncoder (model.safetensors, tokenizer)
└── src/
    ├── __init__.py
    ├── utils.py               # Чтение qrels/topics, потоковая фильтрация корпуса, сборка пар
    ├── prepare_data.py        # Генерация корпуса и обучающих пар
    ├── train_reranker.py      # Дообучение CrossEncoder на PyTorch
    ├── evaluate.py            # Оценка метрик ранжирования (MRR@10, P@1, Top-10 Hit)
    └── infer.py               # Двухэтапный поиск BM25 + CrossEncoder
```

---

## Установка и запуск

### 1. Установка зависимостей

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Скачивание и подготовка данных (MIRACL-RU)

Данные проекта распределены по двум официальным репозиториям на Hugging Face:
* **Разметка запросов (`topics` и `qrels`):** [miracl/miracl (`miracl-v1.0-ru`)](https://huggingface.co/datasets/miracl/miracl/tree/main/miracl-v1.0-ru) — скачайте папки `topics` и `qrels` в `data/raw/`.
* **Текстовый корпус статей (`docs-*.jsonl.gz`):** [miracl/miracl-corpus (`miracl-corpus-v1.0-ru`)](https://huggingface.co/datasets/miracl/miracl-corpus/tree/main/miracl-corpus-v1.0-ru) — скачайте и распакуйте архивы `.jsonl.gz` в папку `data/raw/corpus/`.

После размещения файлов запустите фильтрацию корпуса и сборку обучающих пар:

```powershell
python -m src.prepare_data `
  --corpus-dir data\raw\corpus `
  --topics-train data\raw\topics\topics.miracl-v1.0-ru-train.tsv `
  --qrels-train data\raw\qrels\qrels.miracl-v1.0-ru-train.tsv `
  --topics-dev data\raw\topics\topics.miracl-v1.0-ru-dev.tsv `
  --qrels-dev data\raw\qrels\qrels.miracl-v1.0-ru-dev.tsv `
  --output-dir data\processed
```

### 3. Обучение модели (`CrossEncoder`)

```powershell
python -m src.train_reranker `
  --train-pairs data\processed\train_pairs.jsonl `
  --dev-pairs data\processed\dev_pairs.jsonl `
  --output-dir models\reranker `
  --epochs 5 `
  --batch-size 16 `
  --device cuda
```
*(При отсутствии GPU укажите `--device cpu` и уменьшите `--batch-size` до `4` или `8`.)*

### 4. Оценка качества и инференс

```powershell
# Расчет MRR@10, P@1 и Top-10 Hit на dev-выборке
python -m src.evaluate `
  --model-path models\reranker `
  --qrels-dev data\raw\qrels\qrels.miracl-v1.0-ru-dev.tsv `
  --topics-dev data\raw\topics\topics.miracl-v1.0-ru-dev.tsv `
  --corpus-jsonl data\processed\corpus.jsonl `
  --top-k 10

# Поиск ответа на пользовательский запрос
python -m src.infer `
  --model-path models\reranker `
  --corpus-jsonl data\processed\corpus.jsonl `
  --query "Кто такой Юрий Гагарин?" `
  --top-k 3
```
