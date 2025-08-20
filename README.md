
# 📊 Churn Prediction with Machine Learning

## 📌 Description
Ce projet vise à prédire le **churn des clients** (résiliation d’abonnement) en utilisant des modèles de machine learning.  
Le dataset utilisé est **Telco Customer Churn**, qui contient des informations sur les clients, leurs abonnements, services et historique de paiement.

L’objectif est de comparer plusieurs modèles pour identifier celui qui donne les meilleures performances.

---

## 🗂️ Dataset
- **Nom** : Telco-Customer-Churn.csv  
- **Taille** : ~7 000 clients  
- **Colonnes** : Informations personnelles, services utilisés, facturation, statut du client (churn ou non).  
- **Variable cible** : `Churn` (Yes/No → converti en 1/0).

---

## ⚙️ Prétraitement
- Nettoyage des données (valeurs manquantes, conversion des variables catégorielles).  
- Séparation des données en `X` (features) et `y` (cible).  
- Découpage en **train/test**.  
- Gestion du déséquilibre de classes (ex. via `SMOTE` ou `RandomOverSampler`).  
- Standardisation des données avec **StandardScaler**.  

---

## 🤖 Modèles Testés
Trois modèles de classification ont été comparés :  

1. **Logistic Regression**  
   - Baseline model pour la classification binaire.  
   - Solver utilisé : `liblinear`.  

2. **Random Forest**  
   - Méthode d’ensemble basée sur plusieurs arbres de décision.  
   - Reproductibilité assurée avec `random_state=42`.  

3. **XGBoost**  
   - Algorithme de gradient boosting très performant.  
   - Paramètre : `eval_metric='logloss'`.  

---

## 🔄 Pipeline
Chaque modèle est intégré dans un **pipeline** :  

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),   # Normalisation des données
    ('model', model)                # Modèle ML choisi
])
