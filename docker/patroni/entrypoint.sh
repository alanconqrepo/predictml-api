#!/bin/bash
# docker/patroni/entrypoint.sh
#
# Génère patroni.yml depuis le template avant de lancer Patroni.
#
# Pourquoi un template et pas un fichier statique ?
#   Le nom du nœud (PATRONI_NAME) et l'adresse de connexion
#   (connect_address) diffèrent entre patroni-1 et patroni-2.
#   Plutôt que de maintenir deux fichiers presque identiques,
#   on utilise un seul template avec des variables ${VAR}.
#   La commande envsubst substitue toutes les variables d'environnement
#   présentes dans le template au moment du démarrage.

set -e

TEMPLATE="/etc/patroni/patroni-template.yml"
CONFIG="/tmp/patroni.yml"

if [ ! -f "$TEMPLATE" ]; then
    echo "[patroni-entrypoint] ERREUR : template introuvable : $TEMPLATE" >&2
    exit 1
fi

# Substitution des variables d'environnement dans le template.
# Seules les variables listées explicitement sont substituées pour éviter
# de remplacer accidentellement des ${...} qui appartiennent à Patroni lui-même.
envsubst '${PATRONI_NAME} ${POSTGRES_USER} ${POSTGRES_PASSWORD} ${REPLICATION_PASSWORD}' \
    < "$TEMPLATE" > "$CONFIG"

echo "[patroni-entrypoint] Démarrage du nœud : ${PATRONI_NAME}"
exec patroni "$CONFIG"
