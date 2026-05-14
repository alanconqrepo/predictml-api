# ROADMAP_VX

Contexte: proposition de roadmap fonctionnelle pour `predictml-api`, en restant aligné avec un objectif simple de service de prédiction ML/API et sans transformer le projet en plateforme lourde.

## Priorité haute

1. **Route de santé `/health`** — Facile.
   - Vérifie que l’API tourne, que le modèle est chargé et que les dépendances minimales sont disponibles.
   - C’est la brique la plus utile pour les utilisateurs 80/20 et pour le déploiement MLOps.

2. **Route de métadonnées `/model/info`** — Facile.
   - Retourne le nom du modèle, la version, le type de tâche, les features attendues et la dernière date de chargement.
   - Très utile pour éviter les erreurs côté client et standardiser l’usage de l’API.

3. **Route de schéma d’entrée `/model/schema`** — Facile.
   - Expose le format attendu: champs requis, types, valeurs par défaut, contraintes simples.
   - Réduit les allers-retours de dev et aide à générer automatiquement un formulaire ou un client.

4. **Validation renforcée des requêtes de prédiction `/predict`** — Facile à moyenne.
   - Ajoute des messages d’erreur explicites, contrôle des champs manquants, des types et des bornes.
   - C’est une amélioration très rentable car elle diminue fortement les faux bugs côté utilisateur.

## Priorité moyenne

5. **Route batch `/predict/batch`** — Moyenne.
   - Permet d’envoyer plusieurs observations en une requête et de renvoyer une liste de prédictions.
   - Intéressant pour les usages data science classiques, tout en restant simple à implémenter.

6. **Route probabiliste `/predict_proba`** — Moyenne.
   - Retourne les probabilités ou scores de confiance quand le modèle le permet.
   - Pertinent surtout pour classification, et souvent attendu par les utilisateurs métier.

7. **Paramètre de version de modèle** — Moyenne.
   - Exemple: `model_version=latest` ou `model_version=v1` sur `/predict` ou via une route dédiée.
   - Utile si plusieurs modèles coexistent, sans introduire une vraie complexité de registry.

8. **Route de feedback `/feedback`** — Moyenne.
   - Permet de renvoyer un résultat réel plus tard pour monitorer qualité et dérive.
   - Très utile côté MLOps si l’objectif du projet inclut un minimum de boucle d’amélioration.

## Priorité basse

9. **Route de métriques `/metrics`** — Moyenne à difficile.
   - Expose latence, taux d’erreur, nombre de requêtes, répartition des prédictions.
   - Intéressant pour l’exploitation, mais à faire seulement si le projet veut vraiment couvrir le monitoring.

10. **Route d’historique léger `/predictions/{id}`** — Moyenne.
   - Permet de récupérer une prédiction déjà produite si le projet stocke l’historique.
   - À éviter si le stockage persistant n’existe pas encore, sinon cela ajoute vite de la plomberie.

## Ce que je ne rajouterais pas

- Pas de gestion multi-utilisateur lourde si l’API n’est pas exposée à un vrai besoin d’accès granulaire.
- Pas de pipeline d’entraînement complet dans cette API si l’objectif initial est uniquement la prédiction.
- Pas d’interface d’annotation complexe, de feature store ou d’orchestration avancée tant que les besoins 80/20 ne les justifient pas.

## Ordre recommandé

1. `/health`
2. `/model/info`
3. `/model/schema`
4. Validation renforcée de `/predict`
5. `/predict/batch`
6. `/predict_proba`
7. `model_version`
8. `/feedback`
9. `/metrics`
10. `/predictions/{id}`

## Remarque produit

Si les routes de base présentes dans le repo couvrent déjà `predict`, éventuellement `predict_proba`, l’authentification et une structure de modèle claire, alors il vaut mieux s’arrêter là et n’ajouter que la santé, le schéma et les métadonnées. Pour 80% des utilisateurs, ces ajouts améliorent vraiment l’ergonomie sans faire grossir le projet inutilement.