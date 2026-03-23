#  House Price Prediction

> Прогноз цен на недвижимость с помощью линейной регрессии

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![R2](https://img.shields.io/badge/R%C2%B2-91%25-brightgreen?style=flat-square)

---

## О проекте

Веб-приложение для прогноза цен на жильё на основе реального датасета тайваньского рынка недвижимости (414 объектов). Используется модель **Linear Regression** из scikit-learn с точностью **R? = 91%**.

---

## Интерфейс

| Блок | Описание |
|---|---|
| ?? Параметры | Форма ввода + прогноз цены |
| ?? Зависимость от метро | Интерактивный график Chart.js |
| ?? Точность модели | Scatter plot факт vs прогноз |
| ?? Датасет | Первые 10 строк таблицы |
| ?? Признаки модели | Список использованных признаков |

---

## Датасет

**Real Estate Valuation Dataset** (Тайвань)

| Колонка | Описание |
|---|---|
| X1 | Дата сделки |
| X2 | Возраст дома (лет) |
| X3 | Расстояние до метро (метры) |
| X4 | Количество магазинов рядом |
| X5 | Широта |
| X6 | Долгота |
| **Y** | **Цена за единицу площади — целевая переменная** |

---

## Модель

| Признак | Источник | Тип |
|---|---|---|
| house_age | X2 | Входной |
| mrt_log | log(X3) | Производный |
| stores | X4 | Входной |
| area_mean | X5 + X6 + Y | Средняя цена района |

---

## Запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Обучение модели

```bash
python train.py
```

### 3. Запуск сервера

```bash
python app.py
```

Откройте браузер: **http://localhost:5000**

---

## Структура проекта

```
realestate_ai/
+-- app.py              Flask API
+-- train.py            Обучение модели
+-- test.py             Тест модели в консоли
+-- re_model.pkl        Обученная модель
+-- Real estate.csv     Датасет
+-- requirements.txt    Зависимости
L-- templates/
    +-- index.html      Фронтенд
    L-- scatter.png     График точности
```

---

## Технологии

- **Python** — основной язык
- **Pandas + NumPy** — обработка данных
- **Scikit-learn** — LinearRegression, R?
- **Matplotlib** — визуализация
- **Flask** — REST API
- **Chart.js** — интерактивные графики
