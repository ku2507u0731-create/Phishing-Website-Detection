
from pathlib import Path
import io
import pandas as pd
from flask import Flask, render_template_string, request, send_file
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset_phishing.csv"

df = pd.read_csv(DATA_PATH).drop_duplicates()

FEATURES = [c for c in df.columns if c not in ["url", "status"]]
X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = (df["status"].str.lower() == "phishing").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(random_state=42))
    ]),
    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ))
    ])
}

accuracy = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy[name] = accuracy_score(
        y_test, model.predict(X_test)
    ) * 100

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Phishing Website Detection</title>
<style>
body {
    font-family: Arial;
    background: #eef2f7;
    text-align: center;
    padding: 30px;
}
.container {
    background: white;
    max-width: 750px;
    margin: auto;
    padding: 30px;
    border-radius: 15px;
}
h1 { color: #163b75; }
button {
    background: #1764d1;
    color: white;
    padding: 12px 25px;
    border: none;
    border-radius: 7px;
    cursor: pointer;
}
table {
    width: 100%;
    margin-top: 20px;
    border-collapse: collapse;
}
td, th {
    padding: 12px;
    border: 1px solid #ddd;
}
.note {
    background: #fff3cd;
    padding: 15px;
    margin-top: 20px;
}
</style>
</head>
<body>
<div class="container">
<h1>Phishing Website Detection</h1>
<p>Machine Learning Based Detection System</p>

<h2>Project Details</h2>
<p>Total Dataset Records: {{records}}</p>
<p>Total Features: {{features}}</p>
<p>Algorithms: 4</p>

<h2>Model Accuracy</h2>
<table>
<tr><th>Algorithm</th><th>Accuracy</th></tr>
{% for name, score in accuracy.items() %}
<tr>
<td>{{name}}</td>
<td>{{"%.2f"|format(score)}}%</td>
</tr>
{% endfor %}
</table>

<h2>Upload Feature CSV for Prediction</h2>
<form action="/predict" method="POST" enctype="multipart/form-data">
<select name="model">
{% for name in accuracy %}
<option value="{{name}}">{{name}}</option>
{% endfor %}
</select>
<br><br>
<input type="file" name="file" accept=".csv" required>
<br><br>
<button type="submit">Detect Website</button>
</form>

<div class="note">
<strong>Note:</strong> This project requires 87 numeric
website features. Direct URL scanning is not implemented.
</div>
</div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(
        HTML,
        records=len(df),
        features=len(FEATURES),
        accuracy=accuracy
    )

@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("file")
    model_name = request.form.get("model")

    if not uploaded or model_name not in models:
        return "Invalid file or model selection", 400

    incoming = pd.read_csv(uploaded)

    missing = [c for c in FEATURES if c not in incoming.columns]
    if missing:
        return "CSV is missing required feature columns: " + ", ".join(missing[:10]), 400

    data = incoming[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    predictions = models[model_name].predict(data)

    incoming["Prediction"] = [
        "Phishing" if p == 1 else "Legitimate"
        for p in predictions
    ]

    output = io.BytesIO()
    incoming.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="phishing_predictions.csv"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
