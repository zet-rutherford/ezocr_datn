#!/usr/bin/env bash
export IS_DEBUG=${DEBUG:-false}
# export PYTHONUNBUFFERED=TRUE

exec gunicorn --bind 0.0.0.0:"${SERVER_PORT:-8866}" --timeout 120 \
    --workers "${NUM_WORKERS:-1}" \
    --access-logfile - \
    --threads "${NUM_THREADS:-1}" \
    --error-logfile - \
    run:app