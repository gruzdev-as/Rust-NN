# rust_nn

Учебная реализация нейронной сети на Rust "с нуля" (без фреймворков вроде PyTorch/TensorFlow), обучаемая на MNIST. В репозитории есть две параллельные реализации одной и той же сети:

- **raw** - полностью ручная реализация тензорных операций на `Vec<Vec<f32>>`.
- **nd** - та же архитектура, но поверх [`ndarray`](https://crates.io/crates/ndarray), для сравнения производительности и удобства.

## Структура проекта

```
src/
├── bin/
│   ├── run_nn_raw.rs      # точка входа для "raw" реализации
│   └── run_nn_ndarray.rs  # точка входа для "ndarray" реализации
├── raw/                   # слои, активации, лосс, сеть на чистых Vec
├── nd/                    # то же самое но на ndarray
├── data/                  # Dataset/DataLoader, чтение MNIST PNG, CLI-конфиг
└── metrics/               # accuracy, precision/recall/f-score (macro)
data/mnist_png/            # датасет MNIST в виде PNG поместите сюда (train/ и test/, по папке на класс)
```

## Датасет

Ожидается MNIST в формате PNG-картинок, разложенных по папкам-классам:

```
data/mnist_png/train/<0-9>/*.png
data/mnist_png/test/<0-9>/*.png
```

## Запуск

Сборка проекта:

```bash
cargo build --release
```

Запуск обучения (ndarray-версия):

```bash
cargo run --release --bin run_nn_ndarray
```

Запуск обучения (raw-версия на чистых Vec):

```bash
cargo run --release --bin run_nn_raw
```

По умолчанию используется сеть `784 → 128 → ReLU → 128 → ReLU → 10` (полносвязная, для MNIST), функция потерь Softmax + Cross-Entropy.

### Параметры CLI

Оба бинарника принимают одинаковые флаги:

| Флаг                   | По умолчанию              | Описание                          |
|-------------------------|----------------------------|------------------------------------|
| `--train-data-folder`  | `data/mnist_png/train`     | Путь к обучающей выборке          |
| `--test-data-folder`   | `data/mnist_png/test`      | Путь к тестовой выборке           |
| `--batch-size`         | `64`                       | Размер батча                      |
| `--num-epochs`         | `5`                        | Число эпох                        |
| `--lr`                 | `0.001`                    | Learning rate                     |

Пример:

```bash
cargo run --release --bin run_nn_ndarray -- --num-epochs 10 --batch-size 128 --lr 0.01
```

## Что выводится во время обучения

На каждой эпохе печатается loss по батчам, а после валидации accuracy и macro precision/recall/f-score на тестовой выборке.

