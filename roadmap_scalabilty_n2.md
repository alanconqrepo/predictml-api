# Roadmap Scalabilité — Niveau 2 : Élimination des SPOFs

> **Contexte** : La stack predictml-api (Niveau 1) est déjà robuste pour 80 % des usages
> en production : 3 réplicas API, Redis Sentinel, read replica PostgreSQL, PgBouncer,
> prédictions async via Redis Stream, observabilité complète (Grafana/Prometheus/Loki).
>
> Ce document traite du **Niveau 2** : les trois points de défaillance unique (SPOF —
> _Single Point Of Failure_) qui subsistent et qui, chacun individuellement, peuvent
> provoquer une interruption totale du service — indépendamment de la qualité du reste.

---

## Pourquoi rester self-hosted et sans Kubernetes

### Pas de cloud externe

Déléguer l'infrastructure à AWS, GCP ou Azure résout effectivement les SPOFs (RDS Multi-AZ,
S3, Cloud Load Balancer) — mais introduit des contraintes qui ne conviennent pas à tous les
projets :

- **Coût variable et imprévisible** : une instance RDS Multi-AZ coûte 2–5× une instance simple.
  À volume équivalent, le TCO cloud est supérieur au self-hosted au-delà d'un certain seuil.
- **Dépendance fournisseur (_vendor lock-in_)** : les APIs propriétaires (S3, ElastiCache,
  Aurora) rendent la migration future coûteuse. Les formats ouverts (MinIO, PostgreSQL, Redis)
  restent portables.
- **Contraintes réglementaires** : certains secteurs (santé, défense, finance publique) imposent
  la localisation des données sur des infrastructures maîtrisées. Un hébergeur cloud étranger
  peut être incompatible avec le RGPD ou des exigences de souveraineté.
- **Compétences internes** : opérer du PostgreSQL et du MinIO en interne est une compétence
  standard en data engineering. Opérer du RDS + IAM + VPC + Security Groups est une
  spécialisation DevOps supplémentaire avec un coût d'apprentissage non négligeable.

### Pas de Kubernetes

Kubernetes résout des problèmes réels — auto-scaling réactif, rolling deploys, scheduling
multi-workload — mais son coût opérationnel est significatif :

- **0,5 à 1 ETP** pour maintenir un cluster en production (upgrades, certificats, node pools,
  network policies, RBAC, stockage persistant).
- **Courbe d'apprentissage** : les concepts (pods, services, ingress, PVC, ConfigMap, RBAC,
  Helm, etc.) représentent plusieurs semaines de formation avant d'être opérationnel.
- **Complexité disproportionnée** pour moins de ~20 microservices ou moins de 600 req/s
  soutenus : Docker Compose avec des services bien configurés offre 90 % des bénéfices
  pour 10 % de la complexité.
- **Les SPOFs restent** : Kubernetes ne résout pas MinIO, PostgreSQL ou Nginx de lui-même.
  Il faut toujours Patroni, le mode distribué MinIO, et un load balancer HA — autant de
  choses que l'on peut faire sans Kubernetes.

### L'approche retenue

Corriger les trois SPOFs avec des outils open-source matures, opérables avec les mêmes
compétences Docker que la stack existante, sur la même infrastructure.

---

## Les trois SPOFs

### Vue d'ensemble

```
État actuel (N1) :

  Clients
    │
    ▼
  [Nginx]  ← SPOF 3 : container unique, pas de failover si la machine tombe
    │
    ▼
  [API × 3 réplicas]  ←─ robuste
    │           │
    ▼           ▼
  [PgBouncer]  [Redis Sentinel]  ←─ robuste (HA déjà en place)
    │
    ▼
  [PostgreSQL primary]  ← SPOF 2 : replica existe mais bascule MANUELLE
    │ (réplication streaming)
  [PostgreSQL replica]
    │
  [MinIO]  ← SPOF 1 : instance unique, perte de TOUS les modèles si le disque tombe
```

---

## SPOF 1 — MinIO : stockage des modèles

### Pourquoi c'est critique

MinIO stocke tous les fichiers `.pkl` des modèles ML et tous les scripts `train.py`.
Sans ces fichiers, l'API ne peut plus charger aucun modèle, même si Redis possède encore
un cache chaud — le cache expire (TTL 1h par défaut) et ne peut plus être rechargé.

**Scénarios de panne** :

| Événement | Conséquence |
|---|---|
| Corruption du volume `minio_data` | Perte permanente de tous les modèles |
| Crash du container MinIO | Interruption des prédictions dès que le cache Redis expire |
| Saturation du disque hôte | MinIO refuse les uploads → retrain impossible, nouveaux modèles bloqués |
| Redémarrage du container pendant un upload `.pkl` | Fichier corrompu → `ModelService` retourne une erreur 500 au prochain chargement |

L'instance unique actuelle ne protège contre aucun de ces scénarios. Le `restart: unless-stopped`
redémarre le container mais ne récupère pas un volume corrompu.

### Ce que le mode distribué apporte

MinIO en mode distribué (_Distributed Mode_) répartit les données sur N serveurs (ou volumes)
avec **erasure coding** : chaque objet est découpé en fragments de données et de parité,
répartis sur les N nœuds. Avec 4 nœuds :

- **Tolérance** : 2 nœuds sur 4 peuvent tomber simultanément sans perte de données ni
  interruption du service de lecture.
- **Auto-réparation** : quand un nœud revient, MinIO reconstruit automatiquement les
  fragments manquants depuis les fragments de parité.
- **Cohérence forte** : toutes les écritures sont validées sur le quorum avant d'être
  acquittées — pas d'écriture partielle silencieuse.

Les 4 nœuds peuvent être 4 conteneurs sur la même machine (protection contre la corruption
de volume et les pannes de processus) ou 4 machines distinctes (protection contre la panne
matérielle complète).

### Architecture cible

```
  API / MLflow
      │
      ▼
  [minio-nginx]  ← round-robin entre les 4 nœuds
   /   |   |   \
  m1  m2  m3  m4   ← 4 nœuds MinIO, chacun avec son propre volume
  v1  v2  v3  v4   ← 4 volumes Docker distincts (ou 4 disques physiques)

  Erasure coding 4 nœuds : tolérance à 2 pannes simultanées
```

### Ce qu'il faut faire

**1. Remplacer le service `minio` unique par 4 nœuds dans `docker-compose.yml`**

```yaml
minio-1:
  image: minio/minio:latest
  hostname: minio-1
  command: server http://minio-{1...4}/data --console-address ":9001"
  environment:
    MINIO_ROOT_USER: ${MINIO_ROOT_USER}
    MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
  volumes:
    - minio_data_1:/data
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
    interval: 30s
    timeout: 20s
    retries: 3
  networks:
    - internal

# minio-2, minio-3, minio-4 : même config, hostname et volume différents
```

**2. Ajouter un Nginx de load balancing devant les 4 nœuds**

```yaml
minio-nginx:
  image: nginx:1.27-alpine
  volumes:
    - ./docker/minio-nginx.conf:/etc/nginx/nginx.conf:ro
  networks:
    - internal
  ports:
    - "127.0.0.1:9000:9000"
    - "127.0.0.1:9001:9001"
```

```nginx
# docker/minio-nginx.conf
upstream minio_api {
    least_conn;
    server minio-1:9000;
    server minio-2:9000;
    server minio-3:9000;
    server minio-4:9000;
}
server {
    listen 9000;
    location / {
        proxy_pass         http://minio_api;
        proxy_set_header   Host $host;
        client_max_body_size 500m;  # modèles peuvent être volumineux
    }
}
```

**3. Mettre à jour la variable d'environnement de l'API**

```bash
# .env
MINIO_ENDPOINT=minio-nginx:9000  # au lieu de minio:9000
```

**4. Migrer les données existantes**

```bash
# Avec le client mc (MinIO Client)
mc alias set old http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD
mc alias set new http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD
mc mirror old/models new/models
```

**Effort estimé** : 3–4 heures  
**Risque de migration** : faible — les anciens volumes restent intacts jusqu'à validation

---

## SPOF 2 — PostgreSQL : bascule manuelle en cas de panne du primary

### Pourquoi c'est critique

La stack N1 dispose d'une réplication streaming PostgreSQL (primary → replica) — c'est
bien. Mais la **bascule est entièrement manuelle** : si le primary tombe, il faut qu'un
humain :

1. Détecte la panne (via les alertes Grafana ou un appel d'utilisateur).
2. Se connecte au serveur.
3. Exécute `pg_promote()` sur le replica pour qu'il devienne primary.
4. Reconfigure PgBouncer pour pointer sur le nouveau primary.
5. Redémarre les services.

**Temps de rétablissement typique (RTO) : 15 minutes à plusieurs heures**, selon la
disponibilité de l'ingénieur et l'heure de la panne. Pendant ce temps, toutes les
écritures échouent — l'API retourne des 500 sur `/predict`, les prédictions sont perdues.

**Scénarios de panne** :

| Événement | Conséquence sans Patroni |
|---|---|
| Container `postgres` crash | API en erreur jusqu'à intervention manuelle |
| Corruption des données du primary | Même scénario, potentiellement irréversible |
| OOM kill du process PostgreSQL | Idem — Docker redémarre le container mais les connexions actives sont coupées |
| Panne machine hôte (si multi-VM) | Primary inaccessible, replica disponible mais non promu |

### Ce que Patroni apporte

Patroni est le standard de fait pour la haute disponibilité PostgreSQL. Il s'appuie sur
un **DCS** (_Distributed Configuration Store_, ici **etcd**) comme source de vérité pour
déterminer quel nœud est leader.

Fonctionnement :

1. Patroni tourne en sidecar sur chaque nœud PostgreSQL.
2. Chaque nœud tente de renouveler son bail (_lease_) leader dans etcd toutes les N secondes.
3. Si le primary ne renouvelle plus son bail (panne), etcd invalide le bail.
4. Les replicas détectent l'expiration et déclenchent une **élection**.
5. Le replica le plus à jour est promu primary en ~15–30 secondes.
6. HAProxy, configuré pour interroger l'API REST de Patroni (`GET /primary`), re-route
   automatiquement les nouvelles connexions vers le nouveau primary.

**Ce que cela change concrètement** :

| Avant (N1) | Après (Patroni) |
|---|---|
| Failover manuel : 15 min à plusieurs heures | Failover automatique : 15–30 secondes |
| Dépend de la disponibilité d'un ingénieur | Entièrement automatique, 24h/24 |
| PgBouncer pointe sur une IP fixe | HAProxy route vers le primary élu dynamiquement |
| Risque de split-brain (deux primaries) | Impossible : etcd garantit un unique leader |

### Architecture cible

```
  API
   │
   ▼
[PgBouncer (write)]  →  [HAProxy-Patroni :5432]
                               │          │
                        [Patroni-1]  [Patroni-2]
                        (primary)    (replica)
                               │          │
                        [etcd-1]  [etcd-2]  [etcd-3]
                               (quorum DCS)

  HAProxy interroge GET /primary sur le port REST 8008 de chaque nœud Patroni.
  Seul le primary répond 200 → HAProxy route les writes vers lui.
  Les replicas répondent 200 sur GET /replica → pool de lecture séparé.
```

### Ce qu'il faut faire

**1. Remplacer `postgres` et `postgres-replica` par deux nœuds Patroni**

```yaml
patroni-1:
  image: patroni/patroni:3.3
  environment:
    PATRONI_NAME: patroni-1
    PATRONI_POSTGRESQL_DATA_DIR: /var/lib/postgresql/data
    PATRONI_ETCD3_HOSTS: "etcd-1:2379,etcd-2:2379,etcd-3:2379"
    PATRONI_SUPERUSER_USERNAME: ${POSTGRES_USER:-postgres}
    PATRONI_SUPERUSER_PASSWORD: ${POSTGRES_PASSWORD}
    PATRONI_REPLICATION_USERNAME: replicator
    PATRONI_REPLICATION_PASSWORD: ${REPLICATION_PASSWORD}
    PATRONI_POSTGRESQL_CONNECT_ADDRESS: "patroni-1:5432"
    PATRONI_RESTAPI_CONNECT_ADDRESS: "patroni-1:8008"
  volumes:
    - patroni_data_1:/var/lib/postgresql/data
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8008/health"]
    interval: 10s
    timeout: 5s
    retries: 5
  networks:
    - internal

# patroni-2 : identique, PATRONI_NAME: patroni-2
```

**2. Ajouter un cluster etcd (3 nœuds pour le quorum)**

```yaml
etcd-1:
  image: bitnami/etcd:3.5
  environment:
    ETCD_NAME: etcd-1
    ETCD_INITIAL_CLUSTER: "etcd-1=http://etcd-1:2380,etcd-2=http://etcd-2:2380,etcd-3=http://etcd-3:2380"
    ETCD_INITIAL_CLUSTER_STATE: new
    ETCD_INITIAL_ADVERTISE_PEER_URLS: http://etcd-1:2380
    ETCD_ADVERTISE_CLIENT_URLS: http://etcd-1:2379
    ETCD_LISTEN_PEER_URLS: http://0.0.0.0:2380
    ETCD_LISTEN_CLIENT_URLS: http://0.0.0.0:2379
    ALLOW_NONE_AUTHENTICATION: "yes"
  networks:
    - internal

# etcd-2, etcd-3 : même structure, noms différents
```

**3. Ajouter HAProxy comme routeur dynamique vers le primary Patroni**

```yaml
haproxy-postgres:
  image: haproxy:2.9-alpine
  volumes:
    - ./docker/haproxy-postgres.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
  networks:
    - internal
```

```haproxy
# docker/haproxy-postgres.cfg
frontend postgres_write
    bind *:5432
    default_backend patroni_primary

frontend postgres_read
    bind *:5433
    default_backend patroni_replicas

backend patroni_primary
    option httpchk GET /primary
    http-check expect status 200
    server patroni-1 patroni-1:5432 check port 8008
    server patroni-2 patroni-2:5432 check port 8008

backend patroni_replicas
    balance roundrobin
    option httpchk GET /replica
    http-check expect status 200
    server patroni-1 patroni-1:5432 check port 8008
    server patroni-2 patroni-2:5432 check port 8008
```

**4. Mettre à jour PgBouncer pour pointer sur HAProxy**

```bash
# .env
PGBOUNCER_HOST=haproxy-postgres  # au lieu de postgres
```

**5. Migrer les données existantes**

```bash
# Dump depuis le primary actuel
docker exec predictml-postgres pg_dumpall -U postgres > backup_migration.sql

# Restaurer sur patroni-1 une fois le cluster Patroni démarré
cat backup_migration.sql | docker exec -i <patroni-1-container> psql -U postgres
```

**Effort estimé** : 1 journée (dont ~3h de test du failover)  
**Risque de migration** : moyen — tester impérativement le failover avant de mettre en production

---

## SPOF 3 — Nginx : reverse proxy sans redondance

### Pourquoi c'est (partiellement) moins critique

Contrairement aux deux SPOFs précédents, la panne du container Nginx est déjà partiellement
mitigée :

- `restart: unless-stopped` redémarre le container en quelques secondes si le process crash.
- Les healthchecks sur l'API permettent à Docker de détecter les services en mauvaise santé.

**Ce qui reste un SPOF réel** :

| Événement | `restart: unless-stopped` couvre-t-il ? |
|---|---|
| Process Nginx crash | ✅ Oui — redémarrage en ~5 s |
| Container OOM killed | ✅ Oui — redémarrage immédiat |
| Panne de la machine hôte | ❌ Non — toute la stack tombe |
| Saturation réseau sur l'hôte Nginx | ❌ Non — goulot d'étranglement non distribuable |
| Mise à jour Nginx avec coupure | ❌ Non — pas de rolling update sur container unique |

Ce SPOF ne devient critique que dans deux contextes :
1. La stack est déployée sur **plusieurs machines physiques ou VMs** (l'hôte Nginx peut tomber
   indépendamment des hôtes API).
2. Un **SLA contractuel** impose une disponibilité > 99,9 % (ce qui ne tolère que ~9h de
   coupure par an, panne machine incluse).

Sur une machine unique, ce SPOF est de facto fusionné avec la panne machine — résoudre Nginx
sans multi-machine n'a pas de sens.

### Ce que Keepalived + double Nginx apporte

Keepalived implémente le protocole **VRRP** (_Virtual Router Redundancy Protocol_) entre
deux machines. Une **VIP** (_Virtual IP_) flotte entre les deux hôtes : elle est assignée
au MASTER, et bascule automatiquement sur le BACKUP si le MASTER ne répond plus.

```
DNS → VIP 192.168.1.100 (toujours active)
           │
    ┌──────┴──────┐
    │             │
[Machine A]    [Machine B]
[Nginx]        [Nginx]
[Keepalived    [Keepalived
 MASTER]        BACKUP]
    │
  Trafic normal
```

Si Machine A tombe :
1. Keepalived sur Machine B détecte l'absence de messages VRRP (~2 s).
2. Machine B s'assigne la VIP.
3. Le trafic reprend via Machine B en ~2–3 secondes.
4. Le DNS ne change pas — il pointe toujours sur la VIP.

### Ce qu'il faut faire

Ce fix s'implémente **sur l'hôte**, pas dans Docker. Keepalived interagit avec les
interfaces réseau physiques pour gérer la VIP.

**1. Installer Keepalived sur les deux machines**

```bash
# Sur Machine A et Machine B
apt install -y keepalived
```

**2. Configurer Keepalived sur Machine A (MASTER)**

```bash
# /etc/keepalived/keepalived.conf — Machine A
vrrp_script chk_nginx {
    script "curl -sf http://localhost:80/health || exit 1"
    interval 2
    weight -20      # retire 20 points de priorité si Nginx ne répond pas
    fall 2
    rise 2
}

vrrp_instance VI_PREDICTML {
    state MASTER
    interface eth0             # adapter au nom de l'interface réseau
    virtual_router_id 51
    priority 100
    advert_int 1

    authentication {
        auth_type PASS
        auth_pass ${KEEPALIVED_SECRET}
    }

    virtual_ipaddress {
        192.168.1.100/24       # la VIP — adapter au sous-réseau
    }

    track_script {
        chk_nginx
    }
}
```

**3. Configurer Keepalived sur Machine B (BACKUP)**

```bash
# /etc/keepalived/keepalived.conf — Machine B
# Identique à Machine A, avec deux différences :
state BACKUP
priority 90        # inférieur à Machine A
```

**4. Démarrer et activer Keepalived**

```bash
# Sur les deux machines
systemctl enable keepalived
systemctl start keepalived

# Vérifier que la VIP est bien assignée à Machine A
ip addr show eth0 | grep 192.168.1.100
```

**5. Vérifier le failover**

```bash
# Simuler une panne en stoppant Nginx sur Machine A
docker stop predictml-nginx

# Sur Machine B, vérifier que la VIP a basculé
ip addr show eth0 | grep 192.168.1.100  # doit apparaître sur Machine B

# Vérifier que les requêtes passent toujours
curl http://192.168.1.100/health
```

**Effort estimé** : 3–4 heures (nécessite 2 machines ou VMs configurées)  
**Prérequis** : 2 machines sur le même sous-réseau L2 (même VLAN ou même hôte hyperviseur)

---

## Récapitulatif et ordre d'implémentation

| Priorité | SPOF | Risque actuel | Solution | Effort | Protection obtenue |
|---|---|---|---|---|---|
| **1** | MinIO | Perte définitive de tous les modèles | Mode distribué 4 nœuds | 3–4 h | 2 nœuds sur 4 peuvent tomber |
| **2** | PostgreSQL | Coupure jusqu'à intervention manuelle | Patroni + etcd + HAProxy | 1 jour | Failover automatique en 15–30 s |
| **3** | Nginx | Coupure si la machine hôte tombe | Keepalived + 2 machines | 3–4 h | Bascule VIP en 2–3 s |

### Pourquoi cet ordre

**MinIO en premier** : la perte de données de modèles est silencieuse et irréversible.
Un cache Redis chaud peut masquer la panne pendant 1 heure — après quoi les prédictions
échouent toutes. Le risque de données irrécupérables est le plus élevé.

**Patroni en deuxième** : une panne PostgreSQL primary stoppe immédiatement toutes les
écritures. C'est l'impact le plus visible en production. Le failover manuel peut durer
des heures si la panne survient la nuit ou le week-end.

**Nginx en dernier** : sur machine unique, `restart: unless-stopped` couvre les pannes
de container. Le fix Nginx ne prend tout son sens qu'avec une infrastructure multi-machine,
qui est souvent la condition préalable à la mise en place des deux premiers fixes également.

---

## Ce que cette roadmap N2 ne résout pas

Ces points restent hors périmètre de cette roadmap et constituent le **Niveau 3** :

- **Auto-scaling réactif** : le nombre de réplicas API reste fixe. Absorber un pic de trafic
  × 10 nécessite une intervention manuelle ou Docker Swarm/Kubernetes.
- **Disaster Recovery inter-datacenter** : si l'ensemble de l'infrastructure physique est
  perdue (incendie, inondation), les sauvegardes externes et la réplication géographique
  sont nécessaires. Hors scope self-hosted sur un seul site.
- **Model serving pour modèles volumineux** : au-delà de 50–100 modèles ou de modèles
  > 100 MB (LLM, XGBoost large), Redis n'est plus adapté comme cache de modèles. Un
  model server dédié (Triton Inference Server, BentoML, Ray Serve) devient nécessaire.
- **PostgreSQL write sharding** : au-delà de ~2 000 transactions/seconde en écriture,
  un seul primary PostgreSQL (même avec Patroni) devient le goulot d'étranglement.
  Le partitionnement applicatif ou Citus sont alors à considérer.
