import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna(subset=['age', 'gender', 'ethnicity'])
    for col in ['height', 'weight']:
        df[col] = df[col].fillna(df[col].mean())
    return df

def encode_data(df):
    df = pd.get_dummies(df, columns=['ethnicity'], drop_first=True)
    df['gender_binary'] = df['gender'].map({'M': 1, 'F': 0})
    return df

def split_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test
