
# ICI 2026-04-04 20:00

data = {
        "model_name": "wine_model",
        "features": [13.0, 2.0, 2.4, 20.0, 100.0, 2.0, 2.0, 0.3, 2.0, 5.0, 1.0, 3.0, 800.0]
    }

faut que désormais les predicts accepte aussi ce format la, où "a" et "b" sont les features (nom de colonne dans le df d'apprentissage)

data = {
        "model_name": "xxx",
        "id_obs": "abcd"
        "features": {"a" : 5, "b" : c}
    }

et qu'on stocke id_obs dans la table predictions

ajoute des tests pour ça.

Et mets à jour la documentation

#
. relancer les tests et corriger

#
. Connecter à github, gestion des branch, pull requests etc...



#
. une route pour créer un nouveau modele dans la base postgre
    . penser à ajouter un nouveau champ pour la table model_metadata "mlflow_run_id"    
    qui viendrait d'un scrupt python run_id = mlflow.active_run().info.run_id
    . si "name" du modele existe deja retourner une erreur
     réecrire create_multiple_advanced_models.py pour qu'il fasse appel à cette nouvelle route

. une route pour update une version d'un modele (name+version) :
    .ie modifier les champs 
        "description": null,
        "is_production": true,
        "accuracy": null,
        "features_count": null,
        "classes": null
    . une seule version peut avoir is_production, passer à false l'autre !



. une route pour delete un modele (toutes les version ou une specique) dans postgre + mlflow

. une route pour déposer une nouvelle version d'un modele avec le nouveau "mlflow_run_id" si n'existe pas déja ?
    . c'est à dire un nouveau run dans mlflow où on souhaite register_model
    . se demander comment faire correspondre  model_metadata.version avec la version de model de mlflow ?
    . bien pointer vers la bonne version du modele registered dans mlflow


# Mlflow 
Run = une expérience
un run n'a pas forcément un modele registered dans mlflow..
Model version = une version officielle que tu veux garder / déployer
1 modèle versionné → vient d’un run

#
. revoir create_multiple_models.py  pour qu'il stocke les modèles dans mlflow comme dans create_multiple_advanced_models.py
. Se demander si utile minio_service.upload_model si deja dans minio via mlflow ? si pas le cas retirer cette partie
. et du coup revoir le predict pour aller recupérer le model en pkl stocké via mlflow (mlflow_run_id) et non directement minio


#
. /predict la possibilite d'ajouter  dans le data model_version et point vers le bon modele -> bon run_id de mlflow, sinon prends le is_production à true




# une route create user
# une route get user
# une route delete user


#ICI 2026-04-04 22:00

# relancer le build docker et voir ce qui plante...
  .ImportError: email-validator is not installed, run `pip install 'pydantic[email]'`  when run docker-compose up --build

# ajoute une nouvelle une route get model,  args = name + version et qui retourne les infos de la table model_metadata, si possible charge le model (pkl objject) sinon retourne les infos nécessaires pour le charger en python ?

# une nouvelle route get predictions args de filtrage = name, version (optional), datetieme start end, user (optionnal)

# ajouter user_id_creator a la table model_metadata pour savoir qui a créé le modele/version
 . modifie en conséquence les routes associées get post patch

#init_data/ n'est pas copié dans l'image. Je le lance depuis l'hôte en pointant sur la DB Docker :
 . modifier le volume du docker compose pour que ça soit le cas

# une nouvelle route GET models/name/version

# ICI

# dans la table prections la colonne input_features doit etre un dict de type {'feature1' : , 'feature2' : ,....} et plus une list
 .modifie tout ce qui doit etre modifié y compris init_db et les tests si nécessaire !

# creer une nouvelle table qui contient les données réelles observés :  id_obs model_name observed_result
le but à terme sera de pouvoir comparer les prédictions avec les données observées
une route qui permet d'ajouter une liste [{'id_obs': , , 'model_name', 'observed_result' : ,"user_id"}]
écrase les lignes si 'id_obs': , 'model_name' existe déjà
une route 

# refaire tous les smoke-tests sur le docker en prenant en compte toutes les nouvelles routes, le lancer pour vérifer que tout est ok
  .bien expliquer qui faut lancer docker-compose en amont et la différence avec le dossier /tests via pytetest

# 04/05 09:28
# opentelemetry compatible + grafana connection dans le docker compose pour voir les logs...
# dans le dossier notebooks creer un notebook qui fait l'apprentissage  iris_advanced_model avec usage avancé de mlflow + appel api pour créer nouveau modele/version
    . tous les attributes renseigné
    . plusieurs metrics 
    . plusieurs parameters
    . plusieurs tags 
    . une description
    . evaluation tables logged
    . plusieurs artifacts comme un rapport html avec courbe auc etc...
    . system metices...
    . le code python qui a permis de genere le model...


# creer une application streamlite multipage
 . ajouter dans le docker compose
 . app multipage
 . pouvoir voir les utilisateurs, les administrer, revoke renew api token etc, créer des nouveaux users
 . pouvoir voir les models, les administrer,  passer une version en production , liens vers mlflow run...
 . pourvoir voir les predictions, filtrer par models, date etc...
 . pourvoir avoir des stats utiles sur les predictions : temps de réponses, répartion des prédictions, errors
 . une page qui donne un exemple de code complet mlflow + api pour créer nouveau modele/version

# opentelemetry compatible + grafana connection dans le docker compose pour voir les logs...

 # 05/04 17:25

# Mode Plan : verifier  le plan de tests via pytests (dossier /tests), est ce que ya des choses manquantes ou en doublons corrige et relances

# Mode Plan : lui demander les améliorations qu'on peut faire sur la qualité du code
 . est ce que ya du code mort inutile 
 . des docstrings à ajouter
 . du lint ? 
 . hamoniser regles de codage ?

# error docker au lancement à corriger
 docker logs predictml-mlflow
 (t 5432 failed: FATAL:  database "mlflow" does not exist   => ajouter quelque part de créer la database au démarrage, et mlflow depends de postgre


# refaire la documentation .md avec toutes les nouvelles fonctionnalités/routes schémas des données
 doit contenir des blocs de code python exemple pour appeler les différentes routes
 doit contenir une doc sur le schéma de la base sql, des exemples de requetes, des exemples de code python se connecter à la db et requetes sql
 inclure le README.md 

# Mode Plan : lui demander les améliorations qu'on peut faire en terme de fonctionnalités ? exmple de nouvelles routes ?
 Role Data scientist MLOps engineer, quelles fonctionnalités pourrait être intérréssantes facile à implémenter sans devenir une usine à gaz




# stockage des données d'apprentisage dans minio ? ou via mlflow directement ?

# Mode plan sur la question du  identification du drift (mlops) des features et qualité de predictions
 . tu peux utiliser par exemple les projets : Deepchecks + Evidently
 . où a ton sauvegarder les données d'apprentissage
 . En sachant qu'on a save les predictions, les features, les données observés 

# Mode plan sur la question : interpretability des modeles  : shapvalues, interpretmL
https://github.com/kelvins/awesome-mlops

