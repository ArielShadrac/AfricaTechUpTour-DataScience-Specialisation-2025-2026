"""
Dashboard Steamlit : Smart City Traffic Classification
Auteur : Fildouindé Ariel Shadrac OUEDRAOGO | Mars 2026
Spécialisation Data Science 2025 2026, Africa Tech Up Tour

Lancement :
    streamlit run app_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# CONFIG GLOBALE

st.set_page_config(
    page_title="Smart City Traffic Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")
sns.set_palette("viridis")
CLASSES = ["Low", "Medium", "High"]


# CHARGEMENT ET FEATURE ENGINEERING

@st.cache_data
def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    df["hour"]          = df["Timestamp"].dt.hour
    df["day_of_week"]   = df["Timestamp"].dt.dayofweek
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]      = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["vehicle_rolling_1h"] = (
        df["Vehicle_Count"].rolling(window=12, min_periods=1).mean().shift(1)
    )
    df["speed_rolling_1h"] = (
        df["Traffic_Speed_kmh"].rolling(window=12, min_periods=1).mean().shift(1)
    )
    df["rain_weather_flag"] = (
        df["Weather_Condition"].str.contains("Rain|Storm", na=False).astype(int)
    )
    df["accident_sentiment_interaction"] = (
        df["Accident_Report"] * df["Sentiment_Score"]
    )

    return df.dropna().reset_index(drop=True)

@st.cache_resource
def train_models(df: pd.DataFrame):
    features = [
        "hour", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "Vehicle_Count", "Traffic_Speed_kmh", "Road_Occupancy_%",
        "Sentiment_Score", "Ride_Sharing_Demand", "Parking_Availability",
        "Emission_Levels_g_km", "Energy_Consumption_L_h",
        "Accident_Report", "vehicle_rolling_1h", "speed_rolling_1h",
        "rain_weather_flag", "accident_sentiment_interaction",
        "Weather_Condition", "Traffic_Light_State",
    ]
    num_features = features[:-2]
    cat_features = ["Weather_Condition", "Traffic_Light_State"]

    X = df[features]
    y = df["Traffic_Condition"]

    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ])

    logreg = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    rf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])

    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_lr = logreg.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    # Feature importance
    cat_names = list(
        rf.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(cat_features)
    )
    feature_names = num_features + cat_names
    importances = rf.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(12)

    return {
        "logreg": logreg, "rf": rf,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred_lr": y_pred_lr, "y_pred_rf": y_pred_rf,
        "fi": fi, "features": features,
    }



# SIDEBAR

with st.sidebar:
    st.title("Smart City Traffic")
    st.markdown("**Auteur :** Fildouindé Ariel Shadrac OUEDRAOGO")
    st.markdown("**Africa Tech Up Tour  Mars 2026**")
    st.divider()

    uploaded = st.file_uploader("Charger le dataset (.csv)", type=["csv"])
    st.divider()
    section = st.radio(
        "Navigation",
        [
            "Apercu des donnees",
            "Analyse exploratoire",
            "Feature Engineering",
            "Evaluation des modeles",
            "Prediction en temps reel",
            "Rapport synthetique",
        ],
    )


# CHARGEMENT DES DONNÉES ET ENTRAÎNEMENT DES MODÈLES

if uploaded is None:
    default_path = "smart_mobility_dataset.csv"
    if not os.path.exists(default_path):
        st.warning("Chargez le fichier smart_mobility_dataset.csv dans la barre latérale.")
        st.stop()
    df = load_and_engineer(default_path)
else:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    df = load_and_engineer(tmp_path)

results = train_models(df)



# APERCU DES DONNÉES

if section == "Apercu des donnees":
    st.header("Apercu du dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", f"{df.shape[0]:,}")
    col2.metric("Variables", str(df.shape[1]))
    col3.metric("Debut", str(df["Timestamp"].min().date()))
    col4.metric("Fin", str(df["Timestamp"].max().date()))

    st.subheader("Premieres lignes")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Statistiques descriptives")
    num_cols = [
        "Vehicle_Count", "Traffic_Speed_kmh", "Road_Occupancy_%",
        "Sentiment_Score", "Ride_Sharing_Demand", "Parking_Availability",
        "Emission_Levels_g_km", "Energy_Consumption_L_h",
    ]
    st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)



# ANALYSE EXPLORATOIRE

elif section == "Analyse exploratoire":
    st.header("Analyse exploratoire")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution de la variable cible")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="Traffic_Condition", order=CLASSES, ax=ax)
        ax.set_xlabel("Niveau de congestion")
        ax.set_ylabel("Observations")
        ax.set_title("Distribution Traffic_Condition")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Vitesse selon la congestion")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df, x="Traffic_Condition", y="Traffic_Speed_kmh",
                    order=CLASSES, ax=ax)
        ax.set_xlabel("Niveau de congestion")
        ax.set_ylabel("Vitesse (km/h)")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Occupation de la route selon la congestion")
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df, x="Traffic_Condition", y="Road_Occupancy_%",
                    order=CLASSES, ax=ax)
        ax.set_xlabel("Niveau de congestion")
        ax.set_ylabel("Occupation (%)")
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        st.subheader("Conditions meteo vs congestion")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Weather_Condition", hue="Traffic_Condition", ax=ax)
        ax.set_xlabel("Condition météo")
        ax.set_ylabel("Observations")
        plt.xticks(rotation=40)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Matrice de corrélation")
    num_cols = [
        "Vehicle_Count", "Traffic_Speed_kmh", "Road_Occupancy_%",
        "Sentiment_Score", "Ride_Sharing_Demand", "Parking_Availability",
        "Emission_Levels_g_km", "Energy_Consumption_L_h",
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm",
                fmt=".2f", linewidths=0.4, ax=ax)
    ax.set_title("Matrice de corrélation")
    st.pyplot(fig)
    plt.close(fig)



# FEATURE ENGINEERING

elif section == "Feature Engineering":
    st.header("Feature Engineering")

    fe_cols = [
        "hour", "is_weekend", "hour_sin", "hour_cos",
        "vehicle_rolling_1h", "speed_rolling_1h",
        "rain_weather_flag", "accident_sentiment_interaction",
    ]

    st.subheader("Variables creees")
    descriptions = {
        "hour": "Heure extraite du Timestamp",
        "is_weekend": "1 si samedi ou dimanche, 0 sinon",
        "hour_sin": "Encodage cyclique de l'heure (sinus)",
        "hour_cos": "Encodage cyclique de l'heure (cosinus)",
        "vehicle_rolling_1h": "Moyenne glissante du nb de véhicules (12 obs, shift 1)",
        "speed_rolling_1h": "Moyenne glissante de la vitesse (12 obs, shift 1)",
        "rain_weather_flag": "1 si Weather_Condition contient Rain ou Storm",
        "accident_sentiment_interaction": "Accident_Report × Sentiment_Score",
    }
    desc_df = pd.DataFrame(
        {"Variable": list(descriptions.keys()), "Description": list(descriptions.values())}
    )
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

    st.subheader("Apercu des nouvelles variables")
    st.dataframe(df[fe_cols].head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution horaire du trafic")
        fig, ax = plt.subplots(figsize=(6, 4))
        df_hour = df.groupby(["hour", "Traffic_Condition"]).size().reset_index(name="count")
        sns.lineplot(data=df_hour, x="hour", y="count", hue="Traffic_Condition", ax=ax)
        ax.set_xlabel("Heure de la journée")
        ax.set_ylabel("Nombre d'observations")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Moyennes glissantes (vitesse)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sample = df.iloc[:500]
        ax.plot(sample["Timestamp"], sample["Traffic_Speed_kmh"],
                alpha=0.4, label="Vitesse brute", linewidth=0.8)
        ax.plot(sample["Timestamp"], sample["speed_rolling_1h"],
                label="Rolling 1h", linewidth=1.5)
        ax.set_xlabel("Temps")
        ax.set_ylabel("Vitesse (km/h)")
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)



# EVALUATION DES MODÈLES

elif section == "Evaluation des modeles":
    st.header("Evaluation des modeles")

    y_test     = results["y_test"]
    y_pred_lr  = results["y_pred_lr"]
    y_pred_rf  = results["y_pred_rf"]
    fi         = results["fi"]

    #  Tableau récapitulatif 
    def build_report_df(y_true, y_pred, model_name):
        report = classification_report(y_true, y_pred,
                                       target_names=CLASSES, output_dict=True)
        rows = []
        for cls in CLASSES:
            rows.append({
                "Modele": model_name, "Classe": cls,
                "Precision": round(report[cls]["precision"], 4),
                "Rappel": round(report[cls]["recall"], 4),
                "F1-Score": round(report[cls]["f1-score"], 4),
            })
        rows.append({
            "Modele": model_name, "Classe": "Macro avg",
            "Precision": round(report["macro avg"]["precision"], 4),
            "Rappel": round(report["macro avg"]["recall"], 4),
            "F1-Score": round(report["macro avg"]["f1-score"], 4),
        })
        return rows

    df_metrics = pd.DataFrame(
        build_report_df(y_test, y_pred_lr, "Logistic Regression")
        + build_report_df(y_test, y_pred_rf, "Random Forest")
    )

    st.subheader("Metriques globales")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy - LR", f"{accuracy_score(y_test, y_pred_lr):.4f}")
    col2.metric("F1 Macro - LR", f"{f1_score(y_test, y_pred_lr, average='macro'):.4f}")
    col3.metric("Accuracy - RF", f"{accuracy_score(y_test, y_pred_rf):.4f}")
    col4.metric("F1 Macro - RF", f"{f1_score(y_test, y_pred_rf, average='macro'):.4f}")

    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    #  Graphiques  decomparaison 
    st.subheader("Comparaison visuelle des metriques")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, metric in zip(axes, ["Precision", "Rappel", "F1-Score"]):
        sns.barplot(
            data=df_metrics[df_metrics["Classe"] != "Macro avg"],
            x="Classe", y=metric, hue="Modele",
            palette="viridis", ax=ax,
        )
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(title="Modele", fontsize=8)
    plt.suptitle("Métriques par classe — LR vs RF", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    #  Matrices de confusion 
    st.subheader("Matrices de confusion")
    col1, col2 = st.columns(2)

    with col1:
        cm_lr = confusion_matrix(y_test, y_pred_lr, labels=CLASSES)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title("Logistic Regression")
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vraie valeur")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        cm_rf = confusion_matrix(y_test, y_pred_rf, labels=CLASSES)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title("Random Forest")
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vraie valeur")
        st.pyplot(fig)
        plt.close(fig)

    #  Feature importance 
    st.subheader("Importance des variables (Random Forest Top 12)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title("Top 12 variables les plus importantes")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



# PRÉDICTION EN TEMPS RÉEL

elif section == "Prediction en temps reel":
    st.header("Prediction en temps reel")
    st.markdown("Entrez les valeurs manuellement pour obtenir une prédiction instantanée.")

    col1, col2, col3 = st.columns(3)

    with col1:
        hour              = st.slider("Heure", 0, 23, 8)
        day_of_week       = st.slider("Jour (0=Lundi, 6=Dimanche)", 0, 6, 1)
        vehicle_count     = st.number_input("Nombre de véhicules", 0, 2000, 300)
        speed_kmh         = st.number_input("Vitesse (km/h)", 0, 130, 50)
        road_occupancy    = st.slider("Occupation route (%)", 0, 100, 40)

    with col2:
        sentiment         = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.01)
        ride_demand       = st.number_input("Ride Sharing Demand", 0, 500, 50)
        parking           = st.number_input("Parking Availability", 0, 1000, 200)
        emission          = st.number_input("Emission (g/km)", 0.0, 500.0, 120.0)
        energy            = st.number_input("Energy (L/h)", 0.0, 100.0, 10.0)

    with col3:
        accident          = st.selectbox("Accident signalé", [0, 1])
        weather           = st.selectbox("Météo", df["Weather_Condition"].unique())
        light_state       = st.selectbox("Feu de circulation", df["Traffic_Light_State"].unique())

    is_weekend = int(day_of_week >= 5)
    vehicle_rolling = vehicle_count
    speed_rolling   = speed_kmh

    input_dict = {
        "hour": [hour], "day_of_week": [day_of_week], "is_weekend": [is_weekend],
        "hour_sin": [np.sin(2 * np.pi * hour / 24)],
        "hour_cos": [np.cos(2 * np.pi * hour / 24)],
        "day_of_week_sin": [np.sin(2 * np.pi * day_of_week / 7)],
        "day_of_week_cos": [np.cos(2 * np.pi * day_of_week / 7)],
        "Vehicle_Count": [vehicle_count], "Traffic_Speed_kmh": [speed_kmh],
        "Road_Occupancy_%": [road_occupancy], "Sentiment_Score": [sentiment],
        "Ride_Sharing_Demand": [ride_demand], "Parking_Availability": [parking],
        "Emission_Levels_g_km": [emission], "Energy_Consumption_L_h": [energy],
        "Accident_Report": [accident],
        "vehicle_rolling_1h": [vehicle_rolling], "speed_rolling_1h": [speed_rolling],
        "rain_weather_flag": [int("Rain" in weather or "Storm" in weather)],
        "accident_sentiment_interaction": [accident * sentiment],
        "Weather_Condition": [weather], "Traffic_Light_State": [light_state],
    }
    X_input = pd.DataFrame(input_dict)

    if st.button("Predire", type="primary"):
        model = results["rf"]
        pred  = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        classes_order = model.classes_

        color_map = {"Low": "green", "Medium": "orange", "High": "red"}
        st.markdown(
            f"### Prediction : "
            f"<span style='color:{color_map[pred]};font-weight:bold;font-size:1.4em'>{pred}</span>",
            unsafe_allow_html=True,
        )

        proba_df = pd.DataFrame({"Classe": classes_order, "Probabilite": proba})
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=proba_df, x="Classe", y="Probabilite",
                    palette=["green", "orange", "red"], ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilite")
        ax.set_title("Distribution des probabilités")
        st.pyplot(fig)
        plt.close(fig)



# RAPPORT SYNTHÉTIQUE

elif section == "Rapport synthetique":
    st.header("Rapport synthetique")

    y_test    = results["y_test"]
    y_pred_lr = results["y_pred_lr"]
    y_pred_rf = results["y_pred_rf"]

    acc_lr = accuracy_score(y_test, y_pred_lr)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_lr  = f1_score(y_test, y_pred_lr, average="macro")
    f1_rf  = f1_score(y_test, y_pred_rf, average="macro")
    best   = "Random Forest" if f1_rf >= f1_lr else "Logistic Regression"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset")
        st.markdown(f"- **Observations** : {df.shape[0]:,}")
        st.markdown(f"- **Variables** : {df.shape[1]}")
        st.markdown(f"- **Période** : {df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}")
        st.markdown(f"- **Split** : 80% train / 20% test (chronologique)")

        st.subheader("Feature Engineering")
        st.markdown("- Signaux cycliques heure / jour")
        st.markdown("- Indicateur week-end")
        st.markdown("- Moyennes glissantes 1h (vitesse, véhicules)")
        st.markdown("- Flag pluie/orage")
        st.markdown("- Interaction accident × sentiment")

    with col2:
        st.subheader("Performances")
        st.markdown(f"| Modele              | Accuracy     | F1 Macro |")
        st.markdown(f"| Logistic Regression | {acc_lr:.4f} | {f1_lr:.4f} |")
        st.markdown(f"| Random Forest       | {acc_rf:.4f} | {f1_rf:.4f} |")
        st.success(f"Meilleur modele : **{best}** (F1 macro = {max(f1_lr, f1_rf):.4f})")


    # Sauvegarde du meilleur modèle
    os.makedirs("model", exist_ok=True)
    model_name = "RandomForest" if best == "Random Forest" else "LogisticRegression"
    model_path = f"model/best_model_{model_name}.pkl"
    joblib.dump(results["rf" if best == "Random Forest" else "logreg"], model_path)
    st.success(f"Modele sauvegarde : `{model_path}`")