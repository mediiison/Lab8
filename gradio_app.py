"""
Веб-приложение для демонстрации модели Random Forest
Датасет: Titanic (классификация выживаемости)
Запуск: python gradio_app.py
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── Загрузка и предобработка данных ──────────────────────────
def load_data():
    url = ("https://raw.githubusercontent.com/datasciencedojo/"
           "datasets/master/titanic.csv")
    df = pd.read_csv(url)
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'],
            inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    le = LabelEncoder()
    df['Sex']      = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    return df

df = load_data()
X = df.drop(columns=['Survived'])
y = df['Survived']

# ── Функция обучения ──────────────────────────────────────────
def train_model(n_estimators, max_depth, min_samples_split,
                min_samples_leaf, max_features, test_size):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size / 100,
        random_state=42,
        stratify=y
    )

    max_feat = None if max_features == 'None' else max_features

    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_feat,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred,
                                    target_names=['Не выжил', 'Выжил'])

    # --- Метрики ---
    metrics_text = (
        f"✅ Accuracy:  {acc:.4f}\n"
        f"📊 Train size: {len(X_train)}  |  Test size: {len(X_test)}\n\n"
        f"{report}"
    )

    # --- Матрица ошибок ---
    fig1, ax1 = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
    ax1.set_facecolor('#1e2130')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn',
                xticklabels=['Не выжил', 'Выжил'],
                yticklabels=['Не выжил', 'Выжил'],
                ax=ax1, linewidths=2, linecolor='#1e2130')
    ax1.set_xlabel('Предсказание', color='white')
    ax1.set_ylabel('Факт', color='white')
    ax1.set_title('Матрица ошибок', color='white', fontsize=13,
                  fontweight='bold')
    ax1.tick_params(colors='white')
    fig1.tight_layout()

    # --- ROC-кривая ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
    ax2.set_facecolor('#1e2130')
    ax2.plot(fpr, tpr, color='#7ee8a2', linewidth=2.5,
             label=f'AUC = {roc_auc:.4f}')
    ax2.plot([0, 1], [0, 1], color='#e05c6b',
             linestyle='--', linewidth=1)
    ax2.fill_between(fpr, tpr, alpha=0.1, color='#7ee8a2')
    ax2.set_xlabel('False Positive Rate', color='white')
    ax2.set_ylabel('True Positive Rate', color='white')
    ax2.set_title(f'ROC-кривая (AUC = {roc_auc:.4f})',
                  color='white', fontsize=13, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.spines[:].set_color('#2e3250')
    ax2.legend(facecolor='#1e2130', labelcolor='white')
    fig2.tight_layout()

    # --- Важность признаков ---
    feat_imp = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values()
    fig3, ax3 = plt.subplots(figsize=(7, 4), facecolor='#1e2130')
    ax3.set_facecolor('#1e2130')
    colors_bar = ['#7ee8a2' if v == feat_imp.max()
                  else '#4ecdc4' for v in feat_imp.values]
    bars = ax3.barh(feat_imp.index, feat_imp.values,
                    color=colors_bar, edgecolor='none', height=0.5)
    for bar, val in zip(bars, feat_imp.values):
        ax3.text(val + 0.002,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', color='white', fontsize=10)
    ax3.set_title('Важность признаков', color='white',
                  fontsize=13, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.spines[:].set_visible(False)
    ax3.xaxis.set_visible(False)
    fig3.tight_layout()

    return metrics_text, fig1, fig2, fig3

# ── Функция предсказания ──────────────────────────────────────
def predict_passenger(n_estimators, max_depth, min_samples_split,
                      min_samples_leaf, max_features, test_size,
                      pclass, sex, age, sibsp, parch, fare, embarked):

    max_feat = None if max_features == 'None' else max_features

    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_feat,
        random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    sex_val      = 0 if sex == 'female' else 1
    embarked_val = {'C': 0, 'Q': 1, 'S': 2}[embarked]

    input_data = pd.DataFrame([[
        pclass, sex_val, age, sibsp, parch, fare, embarked_val
    ]], columns=X.columns)

    pred  = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if pred == 1:
        result = f"✅ Пассажир ВЫЖИЛ\n\nВероятность выживания: {proba[1]*100:.1f}%"
    else:
        result = f"❌ Пассажир НЕ ВЫЖИЛ\n\nВероятность гибели: {proba[0]*100:.1f}%"

    # График вероятностей
    fig, ax = plt.subplots(figsize=(5, 2), facecolor='#1e2130')
    ax.set_facecolor('#1e2130')
    ax.barh(['Не выжил', 'Выжил'], proba,
            color=['#e05c6b', '#7ee8a2'],
            edgecolor='none', height=0.4)
    for i, v in enumerate(proba):
        ax.text(v + 0.01, i, f'{v*100:.1f}%',
                va='center', color='white', fontweight='bold')
    ax.set_xlim(0, 1.2)
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_title('Вероятности', color='white', fontsize=11)
    fig.tight_layout()

    return result, fig

# ── Интерфейс Gradio ──────────────────────────────────────────
with gr.Blocks(
    title="🌲 Random Forest Explorer",
    theme=gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate"
    ),
    css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        h1 { text-align: center; color: #7ee8a2 !important; }
        .gr-button-primary { background: #7ee8a2 !important; color: #0f1117 !important; }
    """
) as demo:

    gr.Markdown("""
    # 🌲 Random Forest Explorer
    ### Интерактивная демонстрация модели случайного леса · Датасет Titanic
    """)

    # ── Гиперпараметры (общие для всех вкладок) ───────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Гиперпараметры")
            n_estimators      = gr.Slider(10, 500, value=100, step=10,
                                          label="Число деревьев (n_estimators)")
            max_depth         = gr.Slider(1, 20, value=5, step=1,
                                          label="Максимальная глубина (max_depth)")
            min_samples_split = gr.Slider(2, 20, value=2, step=1,
                                          label="Мин. образцов для разбиения")
            min_samples_leaf  = gr.Slider(1, 10, value=1, step=1,
                                          label="Мин. образцов в листе")
            max_features      = gr.Dropdown(['sqrt', 'log2', 'None'],
                                            value='sqrt',
                                            label="Макс. признаков (max_features)")
            test_size         = gr.Slider(10, 40, value=20, step=5,
                                          label="Размер тестовой выборки, %")
            train_btn         = gr.Button("🚀 Обучить модель",
                                          variant="primary")

        # ── Вкладки результатов ───────────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs():

                # Вкладка 1 — Данные
                with gr.Tab("📊 Данные"):
                    gr.Markdown(f"""
                    **Датасет Titanic** — предсказание выживаемости пассажиров

                    | Параметр | Значение |
                    |---|---|
                    | Записей | {len(df)} |
                    | Признаков | {X.shape[1]} |
                    | Выживших | {y.mean()*100:.1f}% |
                    | Пропусков | 0 (после обработки) |

                    **Признаки:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
                    """)
                    gr.DataFrame(value=df.head(10),
                                 label="Первые 10 строк датасета")

                # Вкладка 2 — Результаты
                with gr.Tab("🎯 Результаты"):
                    metrics_out = gr.Textbox(
                        label="Метрики качества",
                        lines=15,
                        placeholder="Нажмите 'Обучить модель'..."
                    )

                # Вкладка 3 — Графики
                with gr.Tab("📈 Графики"):
                    with gr.Row():
                        cm_plot  = gr.Plot(label="Матрица ошибок")
                        roc_plot = gr.Plot(label="ROC-кривая")
                    fi_plot = gr.Plot(label="Важность признаков")

                # Вкладка 4 — Предсказание
                with gr.Tab("🔍 Предсказание"):
                    gr.Markdown("### Введите параметры пассажира:")
                    with gr.Row():
                        pclass   = gr.Dropdown([1, 2, 3], value=1,
                                               label="Класс билета (Pclass)")
                        sex      = gr.Dropdown(['female', 'male'],
                                               value='female', label="Пол")
                        age      = gr.Slider(1, 80, value=30, label="Возраст")
                    with gr.Row():
                        sibsp    = gr.Slider(0, 8, value=0, step=1,
                                            label="SibSp (братья/сёстры/супруг)")
                        parch    = gr.Slider(0, 6, value=0, step=1,
                                            label="Parch (родители/дети)")
                    with gr.Row():
                        fare     = gr.Slider(0, 512, value=32,
                                            label="Стоимость билета (Fare)")
                        embarked = gr.Dropdown(['C', 'Q', 'S'], value='S',
                                               label="Порт посадки")

                    predict_btn    = gr.Button("🔮 Предсказать", variant="primary")
                    predict_result = gr.Textbox(label="Результат", lines=3)
                    predict_plot   = gr.Plot(label="Вероятности")

    # ── Привязка кнопок ───────────────────────────────────────
    train_btn.click(
        fn=train_model,
        inputs=[n_estimators, max_depth, min_samples_split,
                min_samples_leaf, max_features, test_size],
        outputs=[metrics_out, cm_plot, roc_plot, fi_plot]
    )

    predict_btn.click(
        fn=predict_passenger,
        inputs=[n_estimators, max_depth, min_samples_split,
                min_samples_leaf, max_features, test_size,
                pclass, sex, age, sibsp, parch, fare, embarked],
        outputs=[predict_result, predict_plot]
    )

# ── Запуск ────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=True, debug=False)
