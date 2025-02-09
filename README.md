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
3. Linear Classifier
4. Logistic Regressor
5. SVC
6. DecisionTreeClassifier
7. RandomForestClassifier
8. GradientBoostingClassifier (sklearn)
9. GradientBoosting (написанный ранее)
10. xgboost
11. catboost
12. lightGBM
13. AutoML

cv=5, random_state=42, cross_val_score with roc_auc