quelles sont les améliorations qu'on peut faire en terme de fonctionnalités ? exmple de nouvelles routes ?
mets toi dans le Role Data scientist MLOps engineer, quelles fonctionnalités pourrait être intérréssantes facile à implémenter sans devenir une usine à gaz. garde en tête l'objectif initial du projet ne propose pas des fonctionnalités qui n'ont rien à voir où inutiles pour 80%des utilisateurs. si toutes les fonctionnalités nécessaires sont déjà présentes n'en invente pas de nouvelles. classes tes idées par ordre de priorité et niveau de difficulté à implémenter. écris le résultat dans un nouveau fichier ROADMAP_VX.md toutes les fonctionnalités des roadmap précédentes ont déjà été implémentées. propose les fonctionnalités api et ou dans l'App streamlit utiles pour les utilisateurs. propose uniquement des choses utiles et nécessaires ! si le projet est déjà complet et que tes propositions ne font que complexifié le projet sans apport il vaut mieux éviter.
si tu as quand même des propositions explique en détails le pourquoi et le comment

------------------------------------


en mode plan:
a partir de l'historique des git mettre à jour là documentation présents dans les .md y compris le readme.md pour intégrer les nouvelles fonctionnalités. expliquer aussi comment cloner le projet et lancer docker compose. une doc pour expliquer comment utiliser l'outil pour un débutant avec exemple de code python etc
si certains documents.md de la documentation so t déjà complets et parfait pas nécessité de les modifier . ajouter dans docker compose Anthropic_api_key

------------------------------------

------------------------------------
reflechis en mode plan 
a partir des de deniers PR (une semaine) et de la documentation 
1.ajouter lzs tests unitaires manquants.
2.ajouter les tests d'intégration nécessaire pour valider le bon fonctionnement global du produit
ajouter les tests end to end e2e pour valider le bon fonctionnement globale du produit
détermine quel est le taux de recouvrement des tests avant et après ton plan



est ce que l'intégration avec mlflow est ok à 100%. par exemple est ce que tous les retrain sont stockés dans les expérimentations mlflow avec les bons kpis etc..je souhaite que les fonctionnalités de mlflow soit utilisés à 100%. crée d'implémentation pour que ça soit me cas.




crée une page streamlit d'aide avec chat llm pour aider l'utilisateur d'un point de vue code : génération de script train avec sklearn mlflow..., d'appel api vers la solution, d'aide à l'utilisation de l'App streamlit, explication des différents indicateurs etc... se baser sur la documentation et skills du projet pour répondre, si besoin ajouter de la documentation. possibilité de visualiser directement la doc .md



en tant qu'expert en sécurité analyse la base de code pour identifier problème de sécurités. vérifie qu'aucunz reelle variable d'environnement n'est publié dans git.
crée un plan pour améliorer la sécurité du projet.



crée les tests utiles manquants pour les scripts suivants :
- services/golden_test_service.py
- api/models.py
- api/monitoring.py
- api/prédictions.py
- api/users.py
- db/database.py
- src/main.py

donne le taux de couverture globale suite aux ajouts