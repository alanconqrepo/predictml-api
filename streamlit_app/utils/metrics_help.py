METRIC_HELP = {
    # Erreur / performance
    "taux_erreur": "Part des prédictions incorrectes sur le total. 0 % = aucune erreur, 100 % = toutes incorrectes.",
    "accuracy": "Proportion de prédictions correctes. 1.0 = 100 % de bonnes prédictions.",
    "mae": "Mean Absolute Error : écart moyen absolu entre la prédiction et la valeur réelle. Plus c'est bas, mieux c'est.",
    "rmse": "Root Mean Square Error : écart quadratique moyen. Pénalise davantage les grandes erreurs que le MAE.",
    "r2": "Coefficient de détermination : part de variance expliquée par le modèle. 1.0 = modèle parfait, 0 = modèle nul, négatif = pire que la moyenne.",
    "precision": "Parmi les prédictions positives, proportion réellement positives. Évalue la fiabilité des détections positives. (w.) = moyenne pondérée par classe.",
    "recall": "Parmi les cas positifs réels, proportion correctement détectés. Évalue la capacité à ne pas rater de cas positifs. (w.) = moyenne pondérée par classe.",
    "f1": "Moyenne harmonique de la précision et du rappel. Utile quand les classes sont déséquilibrées. (w.) = moyenne pondérée par classe.",
    # Latence
    "latence_mediane": "Temps de réponse médian : la moitié des requêtes ont un temps inférieur à cette valeur.",
    "latence_avg": "Temps de réponse moyen sur toutes les requêtes.",
    "latence_p95": "95e percentile de latence : 95 % des requêtes sont traitées en moins de ce temps. Indicateur clé de performance en production.",
    # A/B Testing
    "concordance_shadow": "Taux d'accord entre le modèle shadow et le modèle de production : proportion de cas où les deux modèles donnent la même prédiction.",
    "p_value": "Probabilité d'observer un écart aussi grand si les deux versions étaient équivalentes. En dessous du seuil (ex. 0.05) → différence statistiquement significative.",
    "niveau_confiance": "Niveau de confiance du test. 95 % signifie : on accepte 5 % de risque de conclure à tort à une différence alors qu'il n'y en a pas.",
    "test_statistique": "Test statistique utilisé pour comparer les versions. Chi-² pour les taux d'erreur (proportions), Mann-Whitney U pour les distributions de latence (continues).",
    "metrique_analysee": "Métrique sur laquelle porte le test statistique : taux d'erreur ou temps de réponse.",
    # Calibration
    "brier_score": "Brier score : mesure la précision des probabilités prédites. 0 = parfait (probabilités exactes), 1 = pire cas. En dessous de 0.25 est généralement acceptable.",
    "gap_confiance": "Écart entre la confiance moyenne prédite et la précision réelle observée. Positif = surconfiance (le modèle surestime ses certitudes), négatif = sous-confiance. Idéalement proche de 0.",
    "statut_calibration": "Statut de calibration du modèle : OK si l'écart est faible, Surconfiant si le modèle surestime ses probabilités, Sous-confiant s'il les sous-estime.",
    # Confiance
    "confiance_moyenne": "Probabilité maximale moyenne retournée par le modèle pour ses prédictions. Indique à quel point le modèle est 'sûr' en moyenne.",
    "p25_confiance": "25e percentile de confiance : 25 % des prédictions ont une confiance inférieure à cette valeur.",
    "p75_confiance": "75e percentile de confiance : 75 % des prédictions ont une confiance inférieure à cette valeur.",
    # Tendance
    "tendance_performance": "Comparaison de la performance entre la 1re et la 2e moitié de la période sélectionnée. Une baisse indique un drift potentiel du modèle.",
}
