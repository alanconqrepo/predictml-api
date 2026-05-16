# Guide du Dashboard Streamlit — PredictML Admin

Le dashboard est accessible sur **http://localhost:8501**. Connectez-vous avec votre token Bearer.

---

## Connexion

1. Ouvrez http://localhost:8501
2. Dans le champ **API URL**, laissez `http://localhost:8000` (ou remplacez par l'URL de votre serveur)
3. Collez votre **token Bearer** (ex : le token admin par défaut `<ADMIN_TOKEN>`)
4. Cliquez **Se connecter**

Le dashboard détecte automatiquement si votre token est admin ou utilisateur simple.

---

## Page d'accueil

Vue d'ensemble de l'état du système :
- **Santé de l'API** : statut, latence, nombre de modèles en cache
- **Métriques clés** : nombre de modèles actifs, de prédictions, d'utilisateurs
- **Liens rapides** vers tous les services (API Swagger, MLflow, MinIO, Grafana)

---

## 👥 Page 1 — Utilisateurs (`/1_Users`)

**Accessible aux admins uniquement.**

### Ce que vous pouvez faire
- **Créer un utilisateur** : renseignez username, email, rôle (`admin`, `user`, `readonly`) et quota journalier
- **Voir le token** : bouton "Afficher le token" avec copie en un clic
- **Renouveler un token** : bouton "Régénérer" (l'ancien token est immédiatement invalidé)
- **Désactiver / réactiver** un compte
- **Voir les quotas** : jauge de consommation journalière par utilisateur
- **Statistiques d'usage** : expander avec les prédictions par modèle et par jour

### Champs du formulaire de création
| Champ | Description |
|---|---|
| Username | Identifiant unique (pas d'espaces) |
| Email | Adresse email unique |
| Rôle | `admin` = accès total, `user` = prédictions, `readonly` = lecture seule |
| Quota/jour | Nombre max de prédictions par jour (défaut : 1000) |

---

## 🤖 Page 2 — Modèles (`/2_Models`)

### Onglets disponibles

#### Détails
- Tableau de toutes les versions avec accuracy, F1, statut (production/actif/déprécié)
- Bouton **Passer en production** : sélectionnez la version, confirmez
- Télécharger le fichier `.joblib` directement depuis MinIO
- Lien vers le run MLflow associé

#### Uploader un modèle
Formulaire multipart pour uploader un `.joblib` :
- Fichier `.joblib` (obligatoire)
- `train.py` (optionnel — permet le ré-entraînement)
- Métadonnées : name, version, description, algorithm, accuracy, f1_score, features_count, classes

#### What-If Explorer
- Sliders pour chaque feature connue du modèle
- Prédiction en temps réel à chaque modification
- Historique des combinaisons testées

#### Feature Importance (SHAP)
- Importance globale des features sur les N dernières prédictions
- Graphique à barres `mean(|SHAP|)` par feature

#### Validation de schéma
- Testez un JSON de features contre le schéma attendu du modèle
- Retourne : features manquantes, inattendues, coercitions de type

#### Comparaison A/B
- Tableau côte à côte des performances de chaque version
- Résultat du test statistique (Chi-² ou Mann-Whitney U, p-value, winner)

#### Golden Tests
- Voir les cas de test associés au modèle
- Lancer les golden tests sur une version spécifique (PASS/FAIL par cas)

---

## 📊 Page 3 — Prédictions (`/3_Predictions`)

### Onglet Historique
- Filtres : modèle, date de début/fin, statut, version
- Tableau paginé des prédictions avec features détaillées
- Cliquer sur une ligne pour soumettre le résultat observé (feedback)
- Export CSV/JSONL/Parquet des prédictions filtrées

### Onglet Batch
- Soumettez plusieurs prédictions en JSON
- Résultats affichés avec probabilités
- Import/export de résultats observés en CSV

### Purge RGPD
- Simulez (`dry_run=true`) puis confirmez la suppression des prédictions anciennes
- Paramètres : nombre de jours de rétention, modèle cible
- Affiche combien de lignes seraient supprimées avant confirmation

---

## 📈 Page 4 — Stats (`/4_Stats`)

### Métriques globales
- **Volume de prédictions** par heure/jour (graphique temporel)
- **Taux d'erreur** avec alerte si au-dessus du seuil
- **Latence moyenne et P95** par modèle

### Leaderboard
Classement des modèles en production par :
- Accuracy, F1 Score, latence P95, volume de prédictions

### Scatter plot
Accuracy vs Latence P95 pour identifier les compromis performance/vitesse.

### Distribution des prédictions
- Histogramme par classe prédite
- Distribution de confiance (utile pour ajuster `confidence_threshold`)

---

## 💡 Page 5 — Code Example (`/5_Code_Example`)

Exemples de code générés dynamiquement avec vos URL et token de session :
- **Python** : entraîner avec MLflow, uploader, prédire, résultats observés
- **curl / bash** : upload, prédiction, historique, résultat observé
- **JavaScript** : même workflow avec `fetch()`

---

## 🔬 Page 6 — A/B Testing (`/6_AB_Testing`)

### Configuration
- Sélectionnez un modèle et ses versions
- Définissez le `deployment_mode` : `production`, `ab_test`, `shadow`
- Ajustez le `traffic_weight` (0–100% pour les versions en A/B test)

### Résultats
- Tableau comparatif : prédictions, erreurs, latence, taux de concordance (shadow)
- **Test statistique** : p-value, seuil de signification, winner
- Bouton **Promouvoir en production** si le winner est identifié

### Modes de déploiement
| Mode | Comportement |
|---|---|
| `production` | Reçoit 100% du trafic (par défaut) |
| `ab_test` | Reçoit `traffic_weight`% du trafic réel |
| `shadow` | Reçoit toutes les requêtes en background, résultat non retourné |

---

## 🔍 Page 7 — Supervision (`/7_Supervision`)

### Tableau de bord global
- État de santé de chaque modèle en production
- Alertes actives (dérive, taux d'erreur, latence)

### Drift Detection
Pour chaque feature :
- **Z-score** : écart en nombre d'écarts-types par rapport à la baseline
- **PSI** (Population Stability Index) : mesure le décalage de distribution
- **Null rate** : taux de valeurs manquantes
- Statut : `ok`, `warning`, `critical`, `no_baseline`

### Output drift (label shift)
- Compare la distribution des classes prédites à la distribution de référence
- Utile pour détecter un glissement sémantique de la population

### Configuration des seuils
Seuils d'alerte configurables par modèle (surcharge les valeurs globales).

### Export
Rapport exportable en CSV ou Markdown pour partager avec l'équipe.

---

## 🔄 Page 8 — Retrain (`/8_Retrain`)

### Ré-entraînement manuel
1. Sélectionnez le modèle et la version source
2. Renseignez la plage de dates d'entraînement
3. Donnez le numéro de la nouvelle version
4. Optionnel : cochez "Mettre en production automatiquement"
5. Cliquez **Lancer le ré-entraînement**
6. Les logs stdout/stderr s'affichent en temps réel

### Planning automatique (cron)
- Configurez une expression cron (ex : `0 3 * * 1` = chaque lundi à 3h UTC)
- `lookback_days` : fenêtre temporelle des données d'entraînement
- `auto_promote` : promouvoir automatiquement si la politique d'auto-promotion est définie
- Activer/désactiver sans perdre la configuration

### Politique d'auto-promotion
Définissez les critères pour qu'une nouvelle version soit automatiquement promue :
- `min_accuracy` : précision minimale requise
- `max_latency_p95_ms` : latence P95 maximale
- `min_sample_validation` : nombre minimal d'observations de validation
- `min_golden_test_pass_rate` : taux de réussite minimal des golden tests

### Historique des retrains
Tableau chronologique : version source → nouvelle version, métriques avant/après, auto-promue ou non.

---

## 🧪 Page 9 — Golden Tests (`/9_Golden_Tests`)

### Gérer les cas de test
- **Créer** : ajoutez un cas avec features d'entrée + sortie attendue + description
- **Import CSV** : colonnes `input_features` (JSON), `expected_output`, `description`
- **Supprimer** des cas existants

### Exécuter les tests
1. Sélectionnez le modèle et la version à tester
2. Cliquez **Lancer les tests**
3. Chaque cas est marqué **PASS** ou **FAIL** avec diff attendu/reçu

### Intégration avec l'auto-promotion
Configurez `min_golden_test_pass_rate` dans la politique d'auto-promotion pour bloquer la promotion si trop de tests échouent.

---

## 💬 Page 10 — Aide & Assistant IA (`/10_Aide`)

Cette page. Posez vos questions au chatbot Claude spécialisé sur PredictML.

---

## Conseils d'utilisation du dashboard

### Navigation rapide
- La barre latérale Streamlit liste toutes les pages
- L'accueil affiche un tableau récapitulatif avec liens directs

### Actualisation des données
- Beaucoup de données sont mises en cache (TTL 30s)
- Cliquez **🔄 Actualiser** sur les pages qui le proposent pour forcer le rechargement

### Erreurs fréquentes
| Message | Cause | Solution |
|---|---|---|
| "Accès réservé aux administrateurs" | Token non-admin | Utilisez le token admin |
| "Erreur de connexion à l'API" | API non démarrée | `docker-compose up -d api` |
| "Aucun modèle disponible" | DB vide | Uploadez un modèle d'abord |
| "Token invalide" | Token expiré ou incorrect | Reconnectez-vous |
