import torch


def f1_score(y_true, y_pred):
    tp = y_pred[(y_pred == 1) & (y_true == 1)].shape[0]
    fp = y_pred[(y_pred == 1) & (y_true != 1)].shape[0]
    fn = y_pred[(y_pred != 1) & (y_true == 1)].shape[0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)


def accuracy_score(y_true, y_pred):
    return torch.sum(y_true == y_pred) / len(y_true)


def cross_val_score(model, X, y, cv=5, scoring=f1_score, shuffle=True):
    X = torch.tensor(X.values, dtype=torch.float)
    y = torch.tensor(y.values, dtype=torch.float)
    if shuffle:
        perm = torch.randperm(X.shape[0])
        X = X[perm]
        y = y[perm]

    scores = torch.zeros(cv)
    step = X.shape[0] // cv
    for i in range(cv):
        i_start = i * step
        i_end = (i + 1) * step
        if i == 0:
            X_train = X[i_end:]
            Y_train = y[i_end:]
        elif i == cv - 1:
            X_train = X[:i_start]
            Y_train = y[:i_start]
        else:
            X_train = torch.concat([X[:i_start], X[i_end:]], dim=0)
            Y_train = torch.concat([y[:i_start], y[i_end:]], dim=0)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X[i_start:i_end])
        y_pred = torch.tensor(y_pred, dtype=torch.float)
        scores[i] = scoring(y[i_start:i_end], y_pred)
    return scores
