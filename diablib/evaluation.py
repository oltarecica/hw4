from sklearn.metrics import roc_auc_score

def evaluate_model(model, train_df, test_df, features, target):
    train_pred = model.predict_proba(train_df[features])[:, 1]
    test_pred = model.predict_proba(test_df[features])[:, 1]

    train_auc = roc_auc_score(train_df[target], train_pred)
    test_auc = roc_auc_score(test_df[target], test_pred)

    return train_auc, test_auc
