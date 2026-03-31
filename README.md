
# Smart City Traffic Congestion Classification

**Auteur** : Fildouindé ArielShadrac OUEDRAOGO  
**Date** : Mars 2026  
**Formation** : Data Science – Africa Tech Up Tour

## Contexte du projet
Dans le cadre d’une Smart City, la connaissance en temps réel du niveau de congestion routière permet d’optimiser les déplacements, la gestion des feux tricolores et l’information des usagers.  
L’objectif est de développer un modèle de **classification supervisée** capable de prédire le niveau de congestion (`Low`, `Medium`, `High`) à partir de données de mobilité urbaine.

## Structure du dépôt
```
.
├── dataset/
│   └── smart_mobility_dataset.csv        # Jeu de données d'origine
├── models/
│   └── traffic_classifier_RandomForest.pkl  # Meilleur modèle sauvegardé
├── notebooks/
│   └── smart_city_traffic.ipynb          # Notebook complet (EDA, FE, modélisation)
├── app_dashboard.py                       # Dashboard Streamlit interactif
├── rapport_synthétique.txt                # Rapport final au format texte
├── requirements.txt                       # Dépendances Python
└── README.md                              # Ce fichier
```

## Démarrage rapide

### 1. Cloner le dépôt
```bash
git clone https://github.com/votre-utilisateur/smart-city-traffic.git
cd smart-city-traffic
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Exécuter le notebook (optionnel)
```bash
jupyter notebook notebooks/smart_city_traffic.ipynb
```

### 4. Lancer le dashboard Streamlit
```bash
streamlit run app_dashboard.py
```

Le dashboard s’ouvre dans votre navigateur. Vous pouvez :
- Visualiser les données et les analyses exploratoires
- Comparer les performances des modèles (Logistic Regression vs Random Forest)
- Faire des prédictions en temps réel en saisissant les caractéristiques du trafic

## Jeu de données
Le jeu de données `smart_mobility_dataset.csv` contient des enregistrements temporels avec une granularité de 5 minutes. Les principales variables sont :
- **Trafic** : `Vehicle_Count`, `Traffic_Speed_kmh`, `Road_Occupancy_%`, `Traffic_Light_State`
- **Environnement** : `Weather_Condition`, `Emission_Levels_g_km`, `Energy_Consumption_L_h`
- **Comportement** : `Sentiment_Score`, `Ride_Sharing_Demand`, `Parking_Availability`
- **Incidents** : `Accident_Report`

La période couvre plusieurs mois. Après feature engineering, le jeu final compte **4 999 lignes** et **26 colonnes**.

## Méthodologie

### Feature engineering
- Variables temporelles : heure, jour de la semaine, week-end
- Encodages cycliques (sin/cos) de l’heure et du jour
- Moyennes glissantes (fenêtre 1h) de `Vehicle_Count` et `Traffic_Speed_kmh`
- Flag météo `rain_weather_flag`
- Interaction `Accident_Report × Sentiment_Score`

### Prétraitement
- Imputation des valeurs manquantes : médiane pour les numériques, constante `"missing"` pour les catégorielles
- Standardisation des variables numériques
- One‑hot encoding des variables catégorielles (`Weather_Condition`, `Traffic_Light_State`)

### Modélisation
Deux modèles sont entraînés et comparés :
- **Régression logistique multinomiale** (baseline)
- **Random Forest** (200 arbres)

Un **split temporel** (80% train / 20% test) est utilisé pour respecter la chronologie.

## Résultats

| Modèle                | Accuracy | F1 (weighted) |
|-----------------------|----------|---------------|
| Logistic Regression   | 0.7670   | 0.7628        |
| Random Forest         | **0.9980** | **0.9980**   |

Le Random Forest obtient des performances quasi parfaites. Les variables les plus importantes sont :
1. `Road_Occupancy_%`
2. `Traffic_Speed_kmh`
3. `Vehicle_Count`
4. `vehicle_rolling_1h`
5. `speed_rolling_1h`

La matrice de confusion (Random Forest) montre une très bonne séparation des classes.

##  Améliorations possibles
- Supprimer les variables trop corrélées (`Vehicle_Count`, `Traffic_Speed_kmh`) pour tester la robustesse
- Validation croisée temporelle (`TimeSeriesSplit`)
- Optimisation des hyperparamètres (GridSearchCV)
- Interprétabilité avancée avec SHAP
- API de prédiction en temps réel (FastAPI)

##  Licence
Ce projet est réalisé dans le cadre d’une formation. Tous droits réservés.

##  Contact
Fildouindé ArielShadrac OUEDRAOGO – [GitHub](https://github.com/arielshadrac)

