
import os, random, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


SEED = 42
MAX_FEATURES = 15000  # Aumentado de 10000
BATCH = 64  # Aumentado de 32
EPOCHS = 50  # Aumentado de 10
LR = 0.0005  # Reduzido de 0.001
TEST_SPLIT = 0.10
VAL_SPLIT = 0.20

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "bot_detection_data.csv")

# Seeds
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#  FUNÇÕES AUXILIARES 
def extract_features(text, df_row=None):
    """Extrai features adicionais do texto"""
    features = {}
    
    # Features básicas do texto
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)
    features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
    
    # Features de pontuação e símbolos
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['period_count'] = text.count('.')
    features['comma_count'] = text.count(',')
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    # Features de URLs e menções
    features['url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    features['mention_count'] = len(re.findall(r'@\w+', text))
    features['hashtag_count'] = len(re.findall(r'#\w+', text))
    
    # Features de repetição
    words = text.lower().split()
    features['unique_word_ratio'] = len(set(words)) / max(len(words), 1)
    features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))
    
    # Features de números
    features['number_count'] = len(re.findall(r'\d+', text))
    features['number_ratio'] = features['number_count'] / max(features['word_count'], 1)
    
    # Features de emojis (aproximação)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    features['emoji_count'] = len(emoji_pattern.findall(text))
    
    # Features de horário (se disponível)
    if df_row is not None and 'Created At' in df_row:
        try:
            from datetime import datetime
            created_at = pd.to_datetime(df_row['Created At'])
            features['hour'] = created_at.hour
            features['day_of_week'] = created_at.weekday()
            features['is_weekend'] = 1 if created_at.weekday() >= 5 else 0
        except:
            features['hour'] = 12
            features['day_of_week'] = 0
            features['is_weekend'] = 0
    else:
        features['hour'] = 12
        features['day_of_week'] = 0
        features['is_weekend'] = 0
    
    # Features de usuário (se disponível)
    if df_row is not None:
        features['retweet_count'] = df_row.get('Retweet Count', 0)
        features['mention_count_user'] = df_row.get('Mention Count', 0)
        features['follower_count'] = df_row.get('Follower Count', 0)
        features['verified'] = 1 if df_row.get('Verified', False) else 0
        features['has_location'] = 1 if pd.notna(df_row.get('Location', '')) else 0
        features['has_hashtags'] = 1 if pd.notna(df_row.get('Hashtags', '')) else 0
    else:
        features['retweet_count'] = 0
        features['mention_count_user'] = 0
        features['follower_count'] = 0
        features['verified'] = 0
        features['has_location'] = 0
        features['has_hashtags'] = 0
    
    return features

def clean_text(text):
    """Limpeza e normalização do texto"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Normalizar espaços
    text = re.sub(r'\s+', ' ', text)
    
    # Remover caracteres especiais excessivos
    text = re.sub(r'[^\w\s@#.,!?]', ' ', text)
    
    # Normalizar repetições de caracteres
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    return text.strip()

#  1) LER CSV 
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV não encontrado em: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print("Colunas no CSV:", df.columns.tolist())
print("Shape:", df.shape)

text_col = "Tweet"
label_col = "Bot Label"

df = df[[text_col, label_col, 'Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Location', 'Created At', 'Hashtags']].dropna(subset=[text_col, label_col]).reset_index(drop=True)

def norm_label(v):
    s = str(v).strip().lower()
    if s in {"1", "true", "bot", "yes", "y"}: return 1
    if s in {"0", "false", "human", "no", "n", "not bot", "non-bot", "legit", "real"}: return 0
    try: return 1 if int(float(s)) != 0 else 0
    except: return 1 if s == "bot" else 0

df["y"] = df[label_col].apply(norm_label).astype(np.int32)
df["x"] = df[text_col].apply(clean_text)

print("Balanceamento (1=bot):")
print(df["y"].value_counts())

#  2) EXTRAIR FEATURES 
print("Extraindo features adicionais...")
feature_list = []
for idx, row in df.iterrows():
    features = extract_features(row["x"], row)
    feature_list.append(features)

feature_df = pd.DataFrame(feature_list)
print(f"Features extraídas: {len(feature_df.columns)}")

#  3) SPLIT 
df_train_val, df_test = train_test_split(df, test_size=TEST_SPLIT, stratify=df["y"], random_state=SEED)
df_train, df_val = train_test_split(df_train_val, test_size=VAL_SPLIT, stratify=df_train_val["y"], random_state=SEED)

# Split das features também
feature_train_val, feature_test = train_test_split(feature_df, test_size=TEST_SPLIT, random_state=SEED)
feature_train, feature_val = train_test_split(feature_train_val, test_size=VAL_SPLIT, random_state=SEED)

print(f"Train: {df_train.shape} | Val: {df_val.shape} | Test: {df_test.shape}")

#  4) TF-IDF VECTORIZATION 
print("Aplicando TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES, 
    stop_words='english', 
    ngram_range=(1, 3),  # Aumentado de (1,2) para (1,3)
    min_df=2,  # Adicionado
    max_df=0.95,  # Adicionado
    sublinear_tf=True  # Adicionado
)

# Fit no conjunto de treino
X_train_tfidf = vectorizer.fit_transform(df_train["x"]).toarray()
X_val_tfidf = vectorizer.transform(df_val["x"]).toarray()
X_test_tfidf = vectorizer.transform(df_test["x"]).toarray()

# Normalizar features numéricas
scaler = StandardScaler()
X_train_features = scaler.fit_transform(feature_train.values)
X_val_features = scaler.transform(feature_val.values)
X_test_features = scaler.transform(feature_test.values)

# Combinar TF-IDF com features extras
X_train = np.hstack([X_train_tfidf, X_train_features])
X_val = np.hstack([X_val_tfidf, X_val_features])
X_test = np.hstack([X_test_tfidf, X_test_features])

y_train = df_train["y"].values
y_val = df_val["y"].values
y_test = df_test["y"].values

print(f"Shape final: {X_train.shape}")
print(f"Features TF-IDF: {X_train_tfidf.shape[1]}")
print(f"Features extras: {X_train_features.shape[1]}")

#  5) MODELOS MÚLTIPLOS 
print("\n=== TREINANDO MODELOS ===")

# Modelo 1: Rede Neural Melhorada
def build_improved_model(input_dim):
    model = keras.Sequential([
        layers.Dense(1024, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(2, activation='softmax')
    ])
    return model

# Modelo 2: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=SEED,
    n_jobs=-1
)

# Modelo 3: Logistic Regression
lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    random_state=SEED
)

# Modelo 4: SVM
svm_model = SVC(
    C=1.0,
    kernel='rbf',
    probability=True,
    random_state=SEED
)

# Ensemble
ensemble_model = VotingClassifier([
    ('rf', rf_model),
    ('lr', lr_model),
    ('svm', svm_model)
], voting='soft')

#  6) TREINAR MODELOS 
# Treinar ensemble
print("Treinando ensemble...")
ensemble_model.fit(X_train, y_train)

# Treinar rede neural
print("Treinando rede neural...")
nn_model = build_improved_model(X_train.shape[1])

opt = keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)
loss = keras.losses.SparseCategoricalCrossentropy()
metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc")]

nn_model.compile(optimizer=opt, loss=loss, metrics=metrics)

# Class weights
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
tot = len(y_train)
class_weight = {0: tot / (2 * max(neg, 1)), 1: tot / (2 * max(pos, 1))}
print("Class weights:", class_weight)

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_acc", mode="max", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint('best_model.h5', monitor='val_acc', save_best_only=True, mode='max')
]

# Treinar rede neural
history = nn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
    batch_size=BATCH
)

#  7) AVALIAÇÃO 
print("\n=== AVALIAÇÃO DOS MODELOS ===")

# Ensemble
ensemble_pred = ensemble_model.predict(X_test)
ensemble_proba = ensemble_model.predict_proba(X_test)[:, 1]

print("\n--- ENSEMBLE ---")
print(classification_report(y_test, ensemble_pred, digits=4))
try:
    print("AUC:", round(float(roc_auc_score(y_test, ensemble_proba)), 4))
except Exception as e:
    print("AUC não calculada:", e)
print("Matriz de confusão:\n", confusion_matrix(y_test, ensemble_pred))

# Rede Neural
nn_pred_proba = nn_model.predict(X_test, verbose=0)
nn_pred = np.argmax(nn_pred_proba, axis=1)
nn_prob = nn_pred_proba[:, 1]

print("\n--- REDE NEURAL ---")
print(classification_report(y_test, nn_pred, digits=4))
try:
    print("AUC:", round(float(roc_auc_score(y_test, nn_prob)), 4))
except Exception as e:
    print("AUC não calculada:", e)
print("Matriz de confusão:\n", confusion_matrix(y_test, nn_pred))

# Modelo híbrido (média das probabilidades)
hybrid_prob = (ensemble_proba + nn_prob) / 2
hybrid_pred = (hybrid_prob > 0.5).astype(int)

print("\n--- MODELO HÍBRIDO ---")
print(classification_report(y_test, hybrid_pred, digits=4))
try:
    print("AUC:", round(float(roc_auc_score(y_test, hybrid_prob)), 4))
except Exception as e:
    print("AUC não calculada:", e)
print("Matriz de confusão:\n", confusion_matrix(y_test, hybrid_pred))

#  8) INFERÊNCIA RÁPIDA 
print("\n=== INFERÊNCIA RÁPIDA ===")
examples = [
    "Win a free iPhone!!! http://spam",
    "Reading AI papers after coffee.",
    "RT @user: This is amazing! #AI #MachineLearning",
    "Buy now! Limited time offer! Click here: http://fake.com",
    "Just had a great day at the park with my family"
]

for example in examples:
    # Limpar texto
    clean_example = clean_text(example)
    
    # Extrair features
    example_features = extract_features(clean_example)
    example_feature_array = np.array([list(example_features.values())])
    example_feature_scaled = scaler.transform(example_feature_array)
    
    # TF-IDF
    example_tfidf = vectorizer.transform([clean_example]).toarray()
    example_combined = np.hstack([example_tfidf, example_feature_scaled])
    
    # Predições
    ensemble_prob = ensemble_model.predict_proba(example_combined)[0, 1]
    nn_prob = nn_model.predict(example_combined, verbose=0)[0, 1]
    hybrid_prob = (ensemble_prob + nn_prob) / 2
    
    print(f"Ex.: {example[:50]}...")
    print(f"  Ensemble P(bot)={ensemble_prob:.4f}")
    print(f"  Neural Net P(bot)={nn_prob:.4f}")
    print(f"  Híbrido P(bot)={hybrid_prob:.4f}")
    print()

# Salvar modelos
print("Salvando modelos...")
joblib.dump(ensemble_model, 'ensemble_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(scaler, 'scaler.pkl')
nn_model.save('nn_model.h5')
print("Modelos salvos com sucesso!")
