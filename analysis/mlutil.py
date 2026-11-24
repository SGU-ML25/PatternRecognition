"""Bộ phân loại học máy dùng chung (impute -> chuẩn hoá -> mô hình)."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_model(kind, seed=0):
    clf = {
        "logreg": lambda: LogisticRegression(C=1.0, max_iter=3000),
        "svm": lambda: SVC(C=3.0, kernel="rbf", gamma="scale"),
        "rf": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=600, early_stopping=True, random_state=seed),
    }[kind]()
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), clf)


MODELS = ["logreg", "svm", "rf", "mlp"]


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", labels=range(5), zero_division=0))
