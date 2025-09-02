# Credit Scoring Assignment

## Part 1: Implement Gradient Boosting with Decision Trees

### Requirements:
Your implementation should support both classification and regression tasks with the following features:

**Loss Functions:**
- Mean Squared Error (MSE) for regression tasks
- Cross-entropy for classification tasks

**Stopping Criteria:**
- Maximum number of trees
- Maximum tree depth

**Additional Parameter:**
- Learning rate (lr)

**Implementation Guidelines:**
- You may utilize decision trees from scikit-learn
- Test your implementation on simple datasets (e.g., fitting a sine curve for regression)

---

## Part 2: Credit Scoring Model Comparison

### Required Models:
Implement and evaluate the following models on the credit scoring dataset:
1. Naive Bayes
2. K-Nearest Neighbors (KNN)
3. Logistic Regression
4. Support Vector Classifier (SVC)
5. Decision Tree Classifier
6. Random Forest Classifier
7. Scikit-learn's GradientBoostingClassifier
8. Your custom Gradient Boosting implementation (from Part 1)
9. XGBoost
10. CatBoost
11. LightGBM
12. AutoML (e.g., LightAutoML)

### Evaluation Protocol:
- **Primary Metric:** ROC AUC score
- **Validation Method:** 5-fold cross-validation
- **Random Seed:** 42 (for reproducibility)
