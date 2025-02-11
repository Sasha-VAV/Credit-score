# Credit-score

Задание на градиентный бустинг:
Реализовать модель градиентного бустинга на pytorch

Основа - дерево из Sklearn

Параметры, необходимые к реализации:

Функция потерь: 
MSE - regression
Cross-entropy - classification
Критерии остановки:
1. Кол-во деревьев
2. Глубина дерева
Параметр обучения lr

Применить на пет-проекте по кредитному скорингу
1. Naive bayes
2. Knn
3. Logistic Regressor
4. SVC
5. DecisionTreeClassifier
6. RandomForestClassifier
7. GradientBoostingClassifier (sklearn)
8. GradientBoosting (написанный ранее)
9. xgboost
10. catboost
11. lightGBM
12. AutoML

cv=5, random_state=42, cross_val_score with roc_auc