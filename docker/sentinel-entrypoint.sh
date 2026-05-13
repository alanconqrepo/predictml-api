#!/bin/sh
# Generates /tmp/sentinel.conf from REDIS_PASSWORD env var and starts Redis Sentinel.
# REDIS_PASSWORD must be set in the container environment.
set -e

cat > /tmp/sentinel.conf <<EOF
sentinel monitor mymaster redis-master 6379 2
sentinel auth-pass mymaster ${REDIS_PASSWORD}
sentinel down-after-milliseconds mymaster 5000
sentinel failover-timeout mymaster 10000
sentinel parallel-syncs mymaster 1
EOF

exec redis-sentinel /tmp/sentinel.conf
