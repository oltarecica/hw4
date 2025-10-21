from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    'age', 'height', 'weight', 'aids', 'cirrhosis', 'hepatic_failure',
    'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis'
]
TARGET = 'diabetes_mellitus'

def train_model(train_df, model_type='logistic'):
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    return model

def predict(model, df):
    X = df[FEATURES]
    df['predictions'] = model.predict_proba(X)[:, 1]
    return df
