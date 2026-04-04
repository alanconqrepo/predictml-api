

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
    . penser à ajouter un nouveau champ pour la table model_metadata "mlflow_run_id"    run_id = mlflow.active_run().info.run_id
    . si name existe deja retourner une erreur
     réecrire create_multiple_advanced_models.py pour qu'il fasse appel à cette nouvelle route

. une route pour update une version d'un modele (name+version) :
    .ie modifier les champs 
        "description": null,
        "is_production": true,
        "accuracy": null,
        "features_count": null,
        "classes": null
    . une seule peut avoir is_production, passer à false l'autre !



. une route pour delete un modele (toutes les version ou une specique) dans postgre + mlflow

. une route pour déposer une nouvelle version d'un modele avec le nouveau "mlflow_run_id"
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
. /predict la possibilite d'ajouter  dans le data model_version et point vers le bon modele, sinon prends le is_production à true



#
1.Un notebook example qui faire un apprentissage d'un modele sklearn sur un dataframe (avec pipeline de transformation), qui va tracker l'experiment/model, et sauvegarder le modele dans l'api  

# une route create user
# une route get user
# une route delete user

# une route get model name/version

# une route get predictions name version datetieme start end


# lui demander les améliorations qu'on peut faire ? exmple de nouvelles routes ?

# refaire la documentation .md avec les nouvelles fonctionnalités

# refaire les tests

# creer un notebook qui fait l'apprentissage  iris_advanced_model
    . tous les attributes renseigné
    . plusieurs metrics 
    . plusieurs parameters
    . plusieurs tags 
    . une description
    . evaluation tables logged
    . plusieurs artifacts comme un rapport html avec courbe auc etc...
    . system metices...
    . le code python qui a permis de genere le model...

# ajouter user_id a la table model_metadata pour savoir qui a créé le modele/version

# creer une nouvelle table qui contient id_obs model_name observed_result
une route qui permet d'ajouter une liste [{'id_obs': , 'model_name', 'observed_result' : ,"user_id"}]
écrase les lignes si 'id_obs': , 'model_name' existe déjà

# creer une application streamlite multipage
 . ajouter dans le docker compose
 . app multipage
 . pouvoir voir les utilisateurs, les administrer, revoke renew api token etc, créer des nouveaux users
 . pouvoir voir les models, les administrer,  passer une version en production , liens vers mlflow run...
 . pourvoir voir les predictions, filtrer par models, date etc...
 . pourvoir avoir des stats utiles sur les predictions : temps de réponses, répartion des prédictions, errors
 . une page qui donne un exemple de code complet mlflow + api pour créer nouveau modele/version


# mlops identification du drift

Deepchecks + Evidently

# stockage des données d'apprentisage dans minio ? ou via mlflow directement ?


# interpretability : shapvalues, interpretmL
https://github.com/kelvins/awesome-mlops

# opentelemetry compatible + grafana connection pour voir les logs...

