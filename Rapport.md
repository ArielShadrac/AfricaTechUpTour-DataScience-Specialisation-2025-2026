![atut_logo](atut_logo.png)

# Rapport synthétique : Classification du niveau de congestion routière

### Spécilisation datascience 2025 - 2026

## 1. Objectif du projet
L’objectif est de développer un modèle de **classification supervisée** capable de prédire le niveau de congestion routière (`Low`, `Medium`, `High`) à partir de données de mobilité urbaine.  
Ce projet s’inscrit dans une **Smart City**, où la connaissance en temps réel du trafic permet d’optimiser les déplacements, la gestion des feux et l’information des usagers.

## 2. Données utilisées
Le jeu de données `smart_mobility_dataset.csv` contient des enregistrements temporels (timestamp) avec des variables :
- **Trafic** : nombre de véhicules, vitesse moyenne, occupation de la route, état des feux tricolores.
- **Environnement** : conditions météo, niveau d’émissions, consommation énergétique.
- **Comportement** : score de sentiment (réseaux sociaux), demande de covoiturage, disponibilité de stationnement.
- **Incidents** : présence d’accident.

La période couvre plusieurs mois allant du 1er mars 2024 au 18 mars 2024, avec une granularité de 5 minutes (12 points par heure). Après feature engineering et suppression des lignes contenant des valeurs manquantes, le jeu de données final compte **4 999 observations** et **26 colonnes**.

## 3. Méthodologie

### 3.1 Prétraitement
Pour pour le pretraitement nous avcont procedé à :
- La conversion de `Timestamp` en format datetime et le tri chronologique.
- La Création de variables temporelles : heure, jour de la semaine, week-end.
- Une **Imputation** : les valeurs manquantes (NaN) issues des moyennes glissantes et des colonnes d’origine sont traitées :
  - Definition de variables numériques : imputation par la **médiane** (`SimpleImputer(strategy="median")`)
  - Definition de variables catégorielles : imputation par une constante `"missing"` (`SimpleImputer(strategy="constant")`)
- Un encodage de la cible `Traffic_Condition` en valeurs numériques (0=Low, 1=Medium, 2=High) avec `LabelEncoder`.

### 3.2 Feature engineering
Afin d’enrichir le pouvoir prédictif, plusieurs nouvelles caractéristiques ont été créées :
- **Signaux cycliques** : sinus et cosinus de l’heure et du jour de la semaine (pour capturer les cycles temporels).
- **Moyennes glissantes** : sur la fenêtre d’une heure (12 points) pour `Vehicle_Count` et `Traffic_Speed_kmh`, décalées d’un pas (`shift(1)`) pour éviter l’utilisation de la valeur courante.
- **Interaction** : `Accident_Report * Sentiment_Score` pour capturer l’impact conjoint d’un accident et du ressenti.
- **Indicateur météo** : `rain_weather_flag` = 1 si la météo contient “Rain” ou “Storm”.

### 3.3 Modélisation
Pour la modélisation nous avons procédé par :
- **Split temporel** : 80% des premières observations pour l’entraînement, 20% les plus récentes pour le test. Ce choix respecte la chronologie et évite la fuite de données futures.
- **Prétraitement** : standardisation des variables numériques, encodage one‑hot des variables catégorielles (`Weather_Condition`, `Traffic_Light_State`).
- **Modèles testés** :
  - Régression logistique multinomiale (baseline).
  - Forêt aléatoire (Random Forest) avec 200 arbres.

## 4. Résultats

### 4.1 Performances
| Modèle                | Accuracy | F1 (weighted) |
|-----------------------|----------|---------------|
| Régression logistique | 0,7670   | 0,7628        |
| Random Forest         | **0,9980** | **0,9980**   |

Le **Random Forest** obtient des performances quasi parfaites, avec des F1‑score par classe (weighted) de 0,998. Les détails par classe sont présentés dans le tableau ci‑dessous.

### 4.2 Matrice de confusion (Random Forest)
```
             Prédit
            Low   Medium  High
Vrai Low     631    0       0
Vrai Medium  0      68      2
Vrai High    0      0       299
```


### 4.3 Variables les plus importantes
D’après la forêt aléatoire, les cinq variables les plus influentes sont :
1. `Road_Occupancy_%`
2. `Traffic_Speed_kmh`
3. `Vehicle_Count`
4. `vehicle_rolling_1h`
5. `speed_rolling_1h`

Ces variables sont toutes directement liées aux caractéristiques dynamiques du trafic, ce qui confirme la cohérence du modèle.

## 5. Limites et discussions

- **Fuite de données potentielles** : les moyennes glissantes sont calculées sur tout le jeu de données avant le split. Même si un `shift(1)` est appliqué, les valeurs des premières observations du test sont influencées par les dernières observations du train. Dans un contexte de prévision, cette approche est acceptable car elle utilise uniquement l’information passée, mais elle doit être signalée.
- **Sur‑apprentissage** : les variables `Vehicle_Count`, `Traffic_Speed_kmh` et `Road_Occupancy_%` sont extrêmement corrélées avec la cible (par construction du jeu). Le modèle pourrait devenir trop dépendant de ces indicateurs et mal généraliser si ces données sont manquantes ou bruitées. Une suppression de ces colonnes forcerait le modèle à apprendre à partir de signaux plus indirects.
- **Déséquilibre léger** : la classe `High` est légèrement sous‑représentée, mais l’excellent F1‑score montre que cela n’a pas affecté la performance.
- **Validation** : un simple split temporel a été utilisé. Une validation croisée temporelle (`TimeSeriesSplit`) aurait permis une évaluation plus robuste et une meilleure estimation de la généralisation.
- **Imputation** : l’utilisation d’une médiane pour les variables numériques et d’une catégorie “missing” pour les variables catégorielles est simple et efficace, mais des techniques plus sophistiquées (comme `IterativeImputer` ou `KNNImputer`) pourraient être explorées.

## 6. Améliorations possibles
Toute oeuvre n'etant pas parfaite nous, nous listons ainsi un ensemble d'amelioratios possibles.
- **Supprimer les variables trop prédictives** : retirer `Vehicle_Count`, `Traffic_Speed_kmh` et `Road_Occupancy_%` pour évaluer la robustesse du modèle sur des données où ces mesures ne seraient pas disponibles.
- **Ajouter des caractéristiques temporelles** : par exemple, le nombre d’heures depuis le dernier accident, l’état du trafic aux heures précédentes via des fenêtres plus longues (moyennes glissantes sur 2h, 6h, etc.).
- **Optimisation des hyperparamètres** : utiliser `GridSearchCV` avec `TimeSeriesSplit` pour affiner les paramètres du Random Forest (n_estimators, max_depth, min_samples_split).
- **Interprétabilité avancée** : intégrer SHAP pour expliquer les prédictions individuellement et mieux comprendre le comportement du modèle.
- **Déploiement en production** : créer une API (FastAPI) exposant le modèle sauvegardé (`traffic_classifier_RandomForest.pkl`) et l’intégrer à un système de collecte de données IoT.

## 7. Conclusion
Ce projet a permis de développer un modèle de classification extrêmement performant pour la prédiction de la congestion routière, avec un F1‑score de **0,998** obtenu par un Random Forest. Le feature engineering temporel a apporté une plus‑value significative, et l’intégration de l’imputation a permis de traiter correctement les valeurs manquantes. Les limites identifiées (dépendance à des variables très corrélées, absence de validation croisée temporelle) ouvrent des pistes d’amélioration pour un passage en production dans une véritable Smart City.

---

**Auteur** : Fildouindé ArielShadrac OUEDRAOGO  
**Date** : Mars 2026