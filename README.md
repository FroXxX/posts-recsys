# posts-recsys
## Описание проекта
Разработка сервиса рекомендательной системы постов в условной социальной сети на основе контентного подхода.

### Исходные данные
##### Таблица Users
*Cодержит информацию о всех пользователях соц.сети*

| Поле      | Описание                                                              |
|-----------|-----------------------------------------------------------------------|
| age       | Возраст пользователя (в профиле)                                      |
| city      | Город пользователя (в профиле)                                        |
| country   | Страна пользователя (в профиле)                                       |
| exp_group | Экспериментальная группа: некоторая зашифрованная категория           |
| gender    | Пол                                                                   |
| id        | Идентификатор                                                         |
| os        | Операционная система устройства                                       |
| source    | Источник трафика                                                      |

Количество зарегистрированных пользователей: ~163 тысячи

##### Таблица posts
*Содержит информацию о постах: уникальный ID каждого поста с соответствующим ей текстом и тематикой*

| Поле  | Описание      |
|-------|---------------|
| id    | Идентификатор |
| text  | Текст поста   |
| topic | Тематика      |

Количество постов: ~7 тысяч

##### Таблица feed_data
*Содержит информацию о действиях пользователей с указанием времени действия*
| Поле      | Описание                                 |
|-----------|------------------------------------------|
| timestamp | Временная отметка отправленного запроса  |
| user_id   | Иденификатор пользователя                |
| post_id   | Иденификатор поста                       |
| action    | Совершённое действие - просмотр или лайк |
| target    | Поставлен ли лайк **после** просмотра.   |

Количество записей: ~77 миллионов

### Описание задачи

Необходимо создать готовый к интеграции веб-сервис, возвращающий по запросу зарегистрированного
пользователя персонализированную ленту новостей.

##### Параметры запроса
| Поле       | Описание                   |
|------------|----------------------------|
| user_id    | Идентификатор пользователя |
| timestamp  | Время запроса              |
| limit      | Количество постов в ленте  |

##### Параметры ответа (одного поста из ленты)
| Поле  | Описание    |
|-------|-------------|
| id    | id поста    |
| text  | Текст поста |
| topic | Тема поста  | 

##### Технические требования и принятые допущения
1. Время отклика сервиса на 1 запрос не должно превышать 500 милисекунд.
1. Сервис не должен занимать более 2 Гб памяти системы.
1. Набор пользователей и постов фиксирован.

## Описание решения

### Анализ данных
-  Исследуемые данные представлены в промежутке с октября 2021 года по декабрь 2021 года.
-  При проведении EDA было замечено, что размер таблицы с информацией о действиях пользователей составляет 77 млн строк. Для уменьшения объема потребляемой оперативной памяти обучение и измерение качества модели проводилось на выборках по 8.33 и 1.67 млн строк соответственно.
-  Кроме того, имеет место серьезный дисбаланс классов - отрицательных классов в ~7.5 раз больше, чем положительных. Соотношение классов было учтено при обучении модели путем увеличения веса объектов положительных классов.


### Алгоритм решения
Решение представляет собой построение рекомендационной системы на контентной основе. Таким образом датасет для обучения рекомендационной модели был сформирован на основе информации о пользователях и постах.
1. Таблица пользователей сама по себе достаточно информативна, поэтому она вошла в состав обучающей выборки в исходном варианте, не получив никаких дополнительных изменений.

1. Таблица **posts** содержит в себе тексты постов в их исходном виде. Из этого следует, что перед приведением в векторный формат эмбеддингов тексты должны пройти некоторую обработку. Для предвариательной обработки текстов в рамках данного проекта был разработан и применен python-пакет [TextPreprocessor](TextPreprocessor/). Пакет позволяет выполнять параллельную обработку текстов в нескольких процессах и включает в себя следующий пайплайн:
    -  *Удаление html тегов и url-ссылок.*
    -  *Расшифовка смыслового значения эмодзи и символьных смайлов.*
    -  *Приведение текста к нижнему регистру.*
    -  *Приведение различных дат к единому формату записи.*
    -  *Расшифровка смыслового значения некоторых символов (% = percent, $ =  dollar).*
    -  *Расшифровка разговорных аббревиатур (asap = as soon as posible, b4 =  before).*
    -  *Приведение различных чисел в текстовый формат.*
    -  *Удаление лишних пробелов.*
    -  *Избавление от пунктуационных символов и стоп-слов.*
    -  *Лемматизация.*

1. Для извлечения информации из смыслового содержания текста постов была применена модель-трансформер **BERT**. Эмбеддинги текстов получены путем усреднения суммы эмбеддингов слов последних 4-х слоев модели. На основе полученных результатов в векторном пространстве эмбеддингов были сформированны 24 кластера. Каждому тексту был присвоен номер кластера и сопоставлены расстояния до каждого кластера.

1. В качестве самой рекомендационной модели был выбран **СatboostClassifier**. Данный выбор обоснован необходимостью ранжировать полученные моделью предсказания по вероятностям, а также возможностью работы с категориальными признаками без предварительной обработки. Гиперпараметры для обучения итоговой модели были извлечены из результатов оптимизации параметров, проводимой с использованием пакета **Optuna**.




### Реализация сервиса
Сервис реализован с помощью фреймворка FastAPI в виде endpoint "ручки":
1. Принимается GET запрос от пользователя на получение ленты рекомендаций.
2. На основе полученных параметров запроса формируется датасет необходимых признаков указанного пользователя для генерации предсказаний модели.
3. Модель делает предсказания для каждого поста, за исключением тех, что пользователь уже видел. Выбираются результаты с наибольшей вероятностью получения лайка.
4. Сервис возвращает отклик со списком рекомендованных постов.

Таким образом веб-сервис содержит следующее API:

**GET /post/recommendations/**

#### Пример запроса
```
GET /post/recommendations/?id=12445&time=1725880614&limit=2
```

#### Пример ответа
```
[
    {
        "id": 4264,
        "text": "Some text about movie.",
        "topic": "movie"
    },
    {
        "id": 6219,
        "text": "Another text about movie.",
        "topic": "movie"
    }
]
```

#### Коды состояния

По мере выполнения endpoint может возвращать различные коды состояния:

| Код       | Описание                   |
|------------|----------------------------|
| 200        | Запрос выполнен успешно. Возвращается список рекомендованных постов. |
| 404        | Пользователь не найден. Убедитесь, что указанный идентификатор пользователя существует.|

### Итоговые метрики
 Поскольку основной задачей рекомендационной системы является корректное ранжирование постов по вероятности лайка, то в качестве метрики для обучения и оценки качества модели использовался ROC-AUC:
```
Качество на трейне: 0.707
Качество на тесте: 0.679
```


### Выводы


-  Разработан и применен python-пакет для предвариательной обработки текста - **TextPreprocessor**.
-  Построена модель рекомендационной системы, принимающая во внимание контекст постов.
-  Реализован веб-сервис, по GET запросу *id* пользователя возвращающий *limit* рекомендованных постов.
-  Для удобства проверки работы сервиса, был подготовлен Docker образ, запускаемый с помощью docker-compose.
-  Кодстайл сервиса проверен flake8 и pyflakes.

### Пути улучшения полученного результата
Ввиду различных допущений и ограничений, а так же учебного характера проекта, укажем возможные пути улучшения сервиса:
1. Возможен более глубокий feature engineering, например кластеризация пользователей.
1. Обученные модели довольно просты. Не исследованы классические RecSys подходы (коллаборативная фильтрация и т.д.).
1. Кроме того, возможна более тонкая настройка гиперпараметров использовавшегося CatboostClassifier.
1. Возможно улучшение сервиса по сбору и мониторингу метрик - времени отклика под нагрузкой, разделение пользователей по группам, список постов в ленте и т.д.

## Структура репозитория

```

├── data
│   ├── processed
│   │   ├── posts_processed.csv
│   │   └── users_processed.csv
│   │
│   └── raw
│       ├── posts_raw.csv
│       └── users_raw.csv
│
├── models
│   └── catboost_bert_cpu_final.cbm
│
├── notebooks
│   ├── data_observe.ipynb
│   └── model_training.ipynb
│       
├── TextPreprocessor
│   ├── __init__.py
│   ├── date_preprocessor.py
│   ├── text_preprocessor.py
│   └── utils.py
│       
├── web_service
│   └── src
│       ├── main.py
│       ├── models.py
│       ├── router.py
│       └── utils.py
│
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements
```

| Название                                                               | Описание                                                      |
|------------------------------------------------------------------------|---------------------------------------------------------------|
| [data](data/)                                                          | Директория с сырыми и обработанными данными                   |
| [models](models/)                                                      | Директория с обученными моделями                              |
| [notebooks](notebooks/)                                                | Директория с Jupyter-ноутбуками                               |
| [data_observe.ipynb](notebooks/data_observe.ipynb)                     | Ноутбук: Анализ исходных данных                                        |
| [model_training.ipynb](notebooks/model_training.ipynb)                 | Ноутбук: Обработка данных, генерация фичей, оптимизация гиперпараметров и обучение модели |
| [TextPreprocessor](TextPreprocessor/)                                  | Python-пакет для предварительной обработки текста             |
| [web_service](web_service/)                                            | Сервис                                                        |


## Инструкция для запуска
#### Способ 1
`git clone https://github.com/FroXxX/posts-recsys.git` <br />
`cd ./posts-recsys` <br />
`docker-compose up -d` <br />

#### Способ 2
`git clone https://github.com/FroXxX/posts-recsys.git` <br />
`cd ./posts-recsys` <br />
`python.exe -m pip install --upgrade pip` <br />
`pip install -r requirements.txt` <br />
`python.exe web_service/src/main.py` <br />

Сервис доступен по http://127.0.0.1:8081/, где во вкладке `/post/recommendations` можно протестировать запрос на выдачу ленты.
Примеры запросов:
-  http://127.0.0.1:8081/post/recommendations?id=2021&time=1728807583&limit=10
-  http://127.0.0.1:8081/post/recommendations?id=10102&time=1577866783&limit=2
-  http://127.0.0.1:8081/post/recommendations?id=167217&time=1620548383&limit=5