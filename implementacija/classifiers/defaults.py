from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


CLASSIFIERS = {
    "dt": DecisionTreeClassifier(random_state=42),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    "lr": LogisticRegression(max_iter=1000, random_state=42),
    "svm": SVC(kernel="rbf", probability=True, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "gnb": GaussianNB(),
    "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, early_stopping=True, random_state=42),
}

_CLASS_WEIGHT_CLFS = {
    "dt": lambda: DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "rf": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "lr": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "svm": lambda: SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, early_stopping=True, random_state=42),
}


def get_classifier(name):
    if name.endswith("_weighted"):
        base = name.replace("_weighted", "")
        if base == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(scale_pos_weight=1, random_state=42, eval_metric="logloss")
        if base in _CLASS_WEIGHT_CLFS:
            return _CLASS_WEIGHT_CLFS[base]()
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(scale_pos_weight=1, random_state=42, eval_metric="logloss")
    if name in CLASSIFIERS:
        return CLASSIFIERS[name]
    raise ValueError(f"Unknown classifier: {name}")


def get_weighted_classifier_names():
    return [f"{k}_weighted" for k in _CLASS_WEIGHT_CLFS] + ["xgboost_weighted"]
