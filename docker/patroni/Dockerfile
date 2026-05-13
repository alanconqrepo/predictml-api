# docker/patroni/Dockerfile
#
# Image custom : PostgreSQL 16 + Patroni (agent HA).
#
# Pourquoi une image custom et pas une image officielle ?
#   Il n'existe pas d'image officielle Patroni maintenue sur Docker Hub.
#   La solution standard en production est de partir de l'image PostgreSQL
#   officielle et d'y installer Patroni via pip.
#
# Ce que Patroni apporte vs un PostgreSQL nu :
#   - Détection automatique des pannes du primary via le bail etcd
#   - Promotion automatique d'un replica en primary (~15-30 s)
#   - Synchronisation automatique (pg_basebackup) des nouveaux nœuds
#   - API REST sur le port 8008 pour les healthchecks HAProxy
#   - pg_rewind : resynchronise un ancien primary sans rebuild complet

FROM postgres:16-alpine

# Dépendances système :
#   python3, py3-pip : requis par Patroni (écrit en Python)
#   curl             : healthchecks Docker (GET /health sur le port 8008)
#   bash             : scripts entrypoint
#   gettext          : commande envsubst pour la génération du patroni.yml
#   su-exec          : changement d'utilisateur léger (image Alpine)
RUN apk add --no-cache python3 py3-pip curl bash gettext su-exec && \
    pip3 install --break-system-packages \
        'patroni[etcd3]==3.3.3' \
        psycopg2-binary

COPY entrypoint.sh /usr/local/bin/patroni-entrypoint.sh
RUN chmod +x /usr/local/bin/patroni-entrypoint.sh

# Patroni doit tourner en tant que postgres pour avoir les droits
# sur le répertoire de données PostgreSQL (/var/lib/postgresql/data).
USER postgres

# 5432 : PostgreSQL (protocole wire)
# 8008 : API REST Patroni (healthchecks HAProxy, patronictl, monitoring)
EXPOSE 5432 8008

ENTRYPOINT ["/usr/local/bin/patroni-entrypoint.sh"]
