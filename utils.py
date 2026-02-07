import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

def parse_layer(value):
    """Qatlam qiymatini parse qilish"""
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if 'sm' in value or 'cm' in value:
        value = value.replace(' sm', '').replace(' cm', '')
    if '-' in value:
        parts = value.split('-')
        if len(parts) == 2:
            try:
                start = float(parts[0].strip())
                end = float(parts[1].strip())
                return (start + end) / 2
            except ValueError:
                pass
    try:
        return float(value)
    except ValueError:
        return np.nan

def load_and_preprocess(data_url):
    """Ma'lumotlarni yuklash va qayta ishlash"""
    df = pd.read_csv(data_url)
    df.columns = [col.strip() for col in df.columns]
    
    expected_columns = [
        'Qatlam (sm)', 'Mexanik tarkib', 'DNS (%)', 'Tuproq zichligi (g/cm³)',
        'pH', 'EC (mS/cm)', 'N (mg/kg)', 'P (mg/kg)', 'K (mg/kg)', 'Gumus (%)',
        'Mg (mg/kg)', 'S (mg/kg)', 'Zn (mg/kg)', 'Mn (mg/kg)', 'B (mg/kg)',
        'Fe (mg/kg)', 'Cu (mg/kg)', 'Mikroorganizmlar(CFU/g)', 'Ekin'
    ]
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    df['Ekin'] = df['Ekin'].fillna(
        df['Ekin'].mode()[0] if not df['Ekin'].mode().empty else 'Unknown'
    ).astype(str)
    
    if 'Qatlam (sm)' in df.columns:
        df['Qatlam (sm)'] = df['Qatlam (sm)'].apply(parse_layer)
        df['Qatlam (sm)'] = pd.to_numeric(df['Qatlam (sm)'], errors='coerce').fillna(df['Qatlam (sm)'].mean())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    le_dict = {}
    
    for col in categorical_cols:
        if col != 'Ekin' and col != 'Namuna':
            df[col] = df[col].fillna(
                df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            ).astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
    
    le_crop = LabelEncoder()
    df['ekin_encoded'] = le_crop.fit_transform(df['Ekin'])
    
    crop_averages = df.groupby('Ekin')[numeric_cols].mean()
    
    return df, le_dict, le_crop, crop_averages

def train_model(df):
    """Modelni o'rgatish"""
    X = df.drop(['Namuna', 'Ekin', 'ekin_encoded'], axis=1, errors='ignore')
    y = df['ekin_encoded']
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train_res, y_train_res)
    
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    cv_scores = cross_val_score(rf, X_train_res, y_train_res, cv=5, scoring='accuracy')
    
    feature_importances = pd.Series(
        rf.feature_importances_, 
        index=X.columns
    ).sort_values(ascending=False)
    
    metrics = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_acc': cv_scores.mean()
    }
    
    return rf, X.columns.tolist(), metrics, feature_importances