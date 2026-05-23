METRIC_HELP = {
    # Supervision globale — KPIs
    "predictions_prod": (
        "Prédictions production retournées aux clients (is_shadow=False) sur la période. "
        "N'inclut pas les exécutions shadow."
    ),
    "predictions_shadow": (
        "Prédictions silencieuses calculées en arrière-plan (is_shadow=True), "
        "non retournées au client.\n\n"
        "Utilisées pour comparer de nouvelles versions avec la production "
        "sans impacter le trafic réel."
    ),
    "modeles_actifs": (
        "Nombre de modèles distincts ayant au moins une version active (is_active=True) "
        "avec des prédictions sur la période sélectionnée.\n\n"
        "Un modèle est « actif » dès qu'une version est chargeable, "
        "quel que soit son mode de déploiement (Production, A/B, Shadow ou aucun routage).\n\n"
        "À ne pas confondre avec « en production » : un modèle actif peut n'être "
        "qu'en shadow ou uploadé sans être exposé au trafic."
    ),
    "alertes_sante": (
        "Nombre de modèles en alerte sur la période.\n\n"
        "Le statut de santé de chaque modèle est le pire des 4 indicateurs :\n\n"
        "• Taux d'erreur exécution\n"
        "  🟡 ≥ 5 %  ·  🔴 ≥ 10 %\n\n"
        "• Drift features (Z-score + PSI + null rate)\n"
        "  Z-score 🟡 ≥ 2  · 🔴 ≥ 3\n"
        "  PSI 🟡 ≥ 0.1  ·  🔴 ≥ 0.2\n"
        "  Null rate 🟡 écart ≥ 5 pts  ·  🔴 ≥ 15 pts ou > 30 %\n\n"
        "• Drift performance (accuracy 1ère vs 2ème moitié de période)\n"
        "  🟡 baisse ≥ 5 pts  ·  🔴 baisse ≥ 10 pts\n\n"
        "• Drift sortie (PSI distribution des prédictions)\n"
        "  🟡 PSI ≥ 0.1  ·  🔴 PSI ≥ 0.2\n\n"
        "Le statut final retenu est le plus sévère parmi ces quatre.\n"
        "⚪ no_data / no_baseline ne dégrade pas le statut global."
    ),
    # Erreur / performance
    "taux_erreur": (
        "Proportion de requêtes ayant échoué à l'exécution sur la période.\n\n"
        "⚠️ Ce n'est PAS un indicateur de qualité ML : une prédiction peut être "
        "correctement calculée et renvoyée au client, mais fausse sur le plan métier.\n\n"
        "Une erreur d'exécution signifie que l'API n'a pas pu produire de prédiction "
        "(exception serveur, modèle non chargé, timeout, entrée invalide…). "
        "Calculé sur les prédictions production uniquement (hors shadow).\n\n"
        "🟡 Avertissement : ≥ 5 %  ·  🔴 Critique : ≥ 10 %"
    ),
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
    # Calibration — classification
    "brier_score": "Brier score : mesure la précision des probabilités prédites. 0 = parfait (probabilités exactes), 1 = pire cas. En dessous de 0.25 est généralement acceptable.",
    "gap_confiance": "Écart entre la confiance moyenne prédite et la précision réelle observée. Positif = surconfiance (le modèle surestime ses certitudes), négatif = sous-confiance. Idéalement proche de 0.",
    "statut_calibration": "Statut de calibration du modèle : OK si l'écart est faible, Surconfiant si le modèle surestime ses probabilités, Sous-confiant s'il les sous-estime.",
    # Calibration — régression
    "calib_mae": (
        "MAE (Mean Absolute Error) : erreur absolue moyenne entre ŷ prédit et y réel.\n\n"
        "MAE = moyenne(|ŷ − y|)\n\n"
        "Plus c'est bas, mieux c'est. "
        "Même unité que la variable cible."
    ),
    "calib_rmse": (
        "RMSE (Root Mean Square Error) : racine de l'erreur quadratique moyenne.\n\n"
        "RMSE = √(moyenne((ŷ − y)²))\n\n"
        "Pénalise davantage les grandes erreurs que le MAE. "
        "Même unité que la variable cible. Plus c'est bas, mieux c'est."
    ),
    "calib_r2": (
        "R² (coefficient de détermination) : part de variance expliquée par le modèle.\n\n"
        "R² = 1 − SS_res / SS_tot\n"
        "SS_res = Σ(ŷ − y)²,  SS_tot = Σ(y − ȳ)²\n\n"
        "1.0 = modèle parfait · 0 = pas mieux que la moyenne · négatif = pire que la moyenne. "
        "Idéalement > 0.80."
    ),
    "calib_biais": (
        "Biais systématique du modèle : moyenne des résidus (ŷ − y).\n\n"
        "Biais = moyenne(ŷ − y)\n\n"
        "Positif → le modèle sur-estime en moyenne.\n"
        "Négatif → le modèle sous-estime en moyenne.\n"
        "Proche de 0 → pas de biais systématique (bonne calibration).\n\n"
        "Statut : 🟢 OK si |biais relatif| < 10 % de l'écart-type des y observés, "
        "🟡 Sur-estimation ou Sous-estimation sinon."
    ),
    # Confiance
    "confiance_moyenne": "Probabilité maximale moyenne retournée par le modèle pour ses prédictions. Indique à quel point le modèle est 'sûr' en moyenne.",
    "p25_confiance": "25e percentile de confiance : 25 % des prédictions ont une confiance inférieure à cette valeur.",
    "p75_confiance": "75e percentile de confiance : 75 % des prédictions ont une confiance inférieure à cette valeur.",
    # Tendance
    "tendance_performance": "Comparaison de la performance entre la 1re et la 2e moitié de la période sélectionnée. Une baisse indique un drift potentiel du modèle.",
}
