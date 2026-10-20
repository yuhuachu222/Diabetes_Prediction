import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score)

st.set_page_config(page_title='Predicting Diabetes Web App', layout='wide')
st.title('Predicting Diabetes Web App')
st.write('Diabetes classification and single-record prediction (for educational use, not medical diagnosis).')

@st.cache_data
def load_data():
    path = Path(__file__).resolve().parent / 'diabetes.csv'
    return pd.read_csv(path)

try:
    df = load_data()
except FileNotFoundError:
    st.error('找不到 diabetes.csv，請將它與 diabetes.py 放在同一個資料夾。')
    st.stop()

if 'Outcome' not in df.columns:
    st.error('資料集缺少 Outcome 欄位。')
    st.stop()

features = df.drop(columns=['Outcome'])
target = df['Outcome']
if features.isnull().any().any() or target.isnull().any():
    st.error('資料中有缺失值，請先清理資料。')
    st.stop()

st.sidebar.title('Model Selection Panel')
classifier = st.sidebar.selectbox('Classifier', ['Decision Tree', 'Logistic Regression', 'Random Forest'])
train_ratio = st.sidebar.slider('Training data ratio (%)', 50, 90, 70, 5)
st.sidebar.caption(f'Train: {train_ratio}% | Test: {100 - train_ratio}%')

params = {}
if classifier == 'Decision Tree':
    params['criterion'] = st.sidebar.radio('Criterion', ['gini', 'entropy'])
    params['splitter'] = st.sidebar.radio('Splitter', ['best', 'random'])
elif classifier == 'Logistic Regression':
    params['C'] = st.sidebar.number_input('C (regularization)', min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    params['max_iter'] = st.sidebar.slider('Maximum iterations', 100, 1000, 500, 50)
else:
    params['n_estimators'] = st.sidebar.number_input('Number of trees', min_value=10, max_value=5000, value=100, step=10)
    params['max_depth'] = st.sidebar.number_input('Maximum depth', min_value=1, max_value=20, value=5)
    params['bootstrap'] = st.sidebar.radio('Bootstrap', [True, False])

metrics = st.sidebar.multiselect('Charts to show', ['Confusion Matrix', 'ROC Curve'], default=['Confusion Matrix', 'ROC Curve'])

# Cache models by algorithm, parameters, and training ratio. Switching the prediction
# input or returning to a previously trained configuration never refits the model.
@st.cache_resource(show_spinner='Training model...')
def train_model(model_name, params_tuple, ratio):
    config = dict(params_tuple)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, train_size=ratio / 100, random_state=0, stratify=target
    )
    if model_name == 'Decision Tree':
        model = DecisionTreeClassifier(**config, random_state=0)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(**config, random_state=0)
    else:
        model = RandomForestClassifier(**config, random_state=0, n_jobs=-1)
    model.fit(x_train, y_train)
    return model, x_test, y_test

key = (classifier, tuple(sorted(params.items())), train_ratio)
if 'trained_keys' not in st.session_state:
    st.session_state.trained_keys = set()

if st.sidebar.button('Train / Classify'):
    st.session_state.trained_keys.add(key)

if key in st.session_state.trained_keys:
    model, x_test, y_test = train_model(*key)
    y_pred = model.predict(x_test)
    st.subheader(f'{classifier} Results')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Accuracy', f'{accuracy_score(y_test, y_pred):.3f}')
    c2.metric('Precision', f'{precision_score(y_test, y_pred, zero_division=0):.3f}')
    c3.metric('Recall', f'{recall_score(y_test, y_pred, zero_division=0):.3f}')
    c4.metric('F1 Score', f'{f1_score(y_test, y_pred, zero_division=0):.3f}')

    if 'Confusion Matrix' in metrics:
        st.subheader('Confusion Matrix')
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=['Non-Diabetes', 'Diabetes']).plot(ax=ax, cmap='Blues')
        st.pyplot(fig)
        plt.close(fig)

    if 'ROC Curve' in metrics:
        st.subheader('ROC Curve')
        proba = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info('請先在左側按 Train / Classify 訓練目前的模型。')

st.divider()
st.subheader('Single Record Prediction')
st.caption('請輸入一筆資料。按下 Predict 只會使用已訓練模型，不會重新訓練。')
with st.form('single_prediction'):
    columns = st.columns(2)
    values = {}
    for i, feature in enumerate(features.columns):
        with columns[i % 2]:
            default = float(features[feature].median())
            values[feature] = st.number_input(feature, value=default, format='%.4f', key=f'input_{feature}')
    submitted = st.form_submit_button('Predict')

if submitted:
    if key not in st.session_state.trained_keys:
        st.warning('請先按左側 Train / Classify 訓練所選模型。')
    else:
        model, _, _ = train_model(*key)
        sample = pd.DataFrame([values], columns=features.columns)
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        st.subheader('Prediction Result')
        st.success('Prediction: Diabetes (1)' if prediction == 1 else 'Prediction: Non-Diabetes (0)')
        fig, ax = plt.subplots()
        prob_by_class = dict(zip(model.classes_, probabilities))
        ax.pie([prob_by_class.get(0, 0), prob_by_class.get(1, 0)],
               labels=['Non-Diabetes', 'Diabetes'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title(f'{classifier} - Predicted Class Probabilities')
        st.pyplot(fig)
        plt.close(fig)
        st.caption('機率是模型估計值，不代表醫療診斷或實際罹病風險。')

if st.sidebar.checkbox('Show raw data'):
    st.subheader('Diabetes Raw Dataset')
    st.dataframe(df)
