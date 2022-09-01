# Итоговый проект по курсу "Рекомендательные системы"

### Задача: Предсказать будущие покупки пользователей
Целевая метрика - MAP@5 > 20%.

Для решения задачи будем использовать двухуровневую рекомендательную систему:
- Модель первого уровня - выдет список из N рекомендаций для каждого пользователя.
- Модель второго уровня - ранжирует список и выдет топ5.

### Лучшее решение по метрике MAP@5:
1. **0.23368523430592378** - ItemItem(BM25 weighting) + CatBoostClassifier(eval_metric='BrierScore')
2. 0.23296551724137907 - ItemItem(BM25 weighting) + LGBMClassifier(objective="binary")

##### Модель lvl 1
На 1м уровне использована модель ItemItemRecommender обученная на взвешенной bm25 матрице, которая выдет список из 500 рекомендаций для каждого пользователя.
Модель ALS с tfidf и BM25 взвешиванием показала более низкие результаты, прикрутить ALS PySpark, к сожалению, не успел.
Пробовал также комбинировать результаты работы моделей на 1м уровне - итоговый результат оказался хуже.

##### Модель lvl 2
На 2м уровне в лучшем решении использована модель CatBoostClassifier c eval_metric "BrierScore". Данная модель обучается на результатах работы 
модели lvl 1 с добавлением фичей и выдает 5 рекомендаций для каждого пользователя. Модель LightGBMClassifier(objective="binary") показала результат немного хуже,
но время ее обучения существенно меньше - 9 секунд против 12,5 минут СatBoost.

##### Описание данных
1. Код решения содержится в файле "final_project.ipynb".
2. В папке "best_rec_lib" содержатся файлы:
	- "metrics.py" - метрики,
	- "utils.py" - обрабока данных, создание фичей,
	- "recommenders.py" - класс рекомендера с моделями.
4. В файле "predictions.csv" находятся итоговые предсказания.

##### Примечание
В рекомендации на lvl 2 важно сделать проверку на уникальность элементов, иначе метрика MAP@5 будет зашкаливать, но по факту это из-за множества дублей.
Причем сделать это надо не через set(), иначе нарушится порядок элементов рекомендации, которого добивались ранжируя данные моделью второго уровня.
