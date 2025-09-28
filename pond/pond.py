import os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow import keras

SEED       = 42
MAX_FEATURES = 10000  
BATCH      = 32
EPOCHS     = 10
LR         = 0.001
TEST_SPLIT = 0.10
VAL_SPLIT  = 0.20

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "bot_detection_data.csv")

# Seeds
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

#  1) LER CSV 
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV não encontrado em: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print("Colunas no CSV:", df.columns.tolist())
print("Shape:", df.shape)

text_col  = "Tweet"
label_col = "Bot Label"

df = df[[text_col, label_col]].dropna().rename(columns={text_col:"text", label_col:"label"}).reset_index(drop=True)

def norm_label(v):
    s = str(v).strip().lower()
    if s in {"1","true","bot","yes","y"}: return 1
    if s in {"0","false","human","no","n","not bot","non-bot","legit","real"}: return 0
    try: return 1 if int(float(s))!=0 else 0
    except: return 1 if s=="bot" else 0

df["y"] = df["label"].apply(norm_label).astype(np.int32)
df["x"] = df["text"].astype(str).str.replace(r"\s+"," ",regex=True).str.strip()

print("Balanceamento (1=bot):")
print(df["y"].value_counts())

#  2) SPLIT 
df_train_val, df_test = train_test_split(df, test_size=TEST_SPLIT, stratify=df["y"], random_state=SEED)
df_train, df_val = train_test_split(df_train_val, test_size=VAL_SPLIT, stratify=df_train_val["y"], random_state=SEED)
print(f"Train: {df_train.shape} | Val: {df_val.shape} | Test: {df_test.shape}")

#  3) TF-IDF VECTORIZATION 
print("Aplicando TF-IDF...")
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english', ngram_range=(1, 2))

# Fit no conjunto de treino
X_train = vectorizer.fit_transform(df_train["x"]).toarray()
X_val = vectorizer.transform(df_val["x"]).toarray()
X_test = vectorizer.transform(df_test["x"]).toarray()

y_train = df_train["y"].values
y_val = df_val["y"].values
y_test = df_test["y"].values

print(f"Shape TF-IDF: {X_train.shape}")
print(f"Features selecionadas: {X_train.shape[1]}")

#  4) MODELO (Neural Network) 
# Cria uma rede neural simples
def build_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation='softmax')
    ])
    return model

classifier = build_model(X_train.shape[1])

opt = keras.optimizers.Adam(learning_rate=LR)
loss = keras.losses.SparseCategoricalCrossentropy()
metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc")]

classifier.compile(optimizer=opt, loss=loss, metrics=metrics)

# Class weights 
pos = int((df_train["y"]==1).sum()); neg = int((df_train["y"]==0).sum()); tot = len(df_train)
class_weight = {0: tot/(2*max(neg,1)), 1: tot/(2*max(pos,1))}
print("Class weights:", class_weight)

early = keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=2, restore_best_weights=True)

#  5) TREINO 
history = classifier.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[early],
    verbose=1,
    batch_size=BATCH
)

#  6) AVALIAÇÃO 
# Predições no conjunto de teste
y_pred_proba = classifier.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_prob = y_pred_proba[:, 1]  # probabilidade da classe 1 (bot)

print("\n=== TEST REPORT ===")
print(classification_report(y_test, y_pred, digits=4))
try:
    print("AUC:", round(float(roc_auc_score(y_test, y_prob)), 4))
except Exception as e:
    print("AUC não calculada:", e)
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

# 7) INFERÊNCIA RÁPIDA 
examples = [
    "Win a free iPhone!!! http://spam",
    "Reading AI papers after coffee."
]
# Transforma os exemplos usando o mesmo vectorizer
examples_tfidf = vectorizer.transform(examples).toarray()
predictions_ex = classifier.predict(examples_tfidf, verbose=0)
p = predictions_ex[:, 1]  # probabilidade da classe 1 (bot)
for t, pp in zip(examples, p):
    print(f"Ex.: {t[:60]}... -> P(bot)={pp:.4f}")
