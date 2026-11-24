#!/usr/bin/env bash
set -e
USERS=${USERS:-100}; SPAWN=${SPAWN:-10}; DURATION=${DURATION:-60s}
mkdir -p results
for N in 1 2 3; do
  echo "=== $N API container(s) ==="
  docker compose up -d --build --scale api=$N
  docker compose restart nginx
  until curl -sf http://localhost:8080/health > /dev/null; do sleep 2; done
  sleep 20   # let the model load in every container
  locust -f locustfile.py --headless -u $USERS -r $SPAWN -t $DURATION \
         --host http://localhost:8080 --csv results/containers_$N --only-summary
done
