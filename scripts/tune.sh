#!/usr/bin/env bash

set -euxo pipefail

export PYTHONUNBUFFERED="1"
export FOR_DISABLE_CONSOLE_CTRL_HANDLER="1"

export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="5QTj8QznwgP##soC"
# export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING="true"

python autogbifml --jobs 12 tune ./dataset/merged/train.parquet ./dataset/models/xgboost_tune --name thesis_xgboost --algorithm xgboost --shuffle --tracking-url http://10.20.20.102:8009 --storage sqlite:///tune-terakhir.db
python autogbifml --jobs 12 tune ./dataset/merged/train.parquet ./dataset/models/catboost_tune --name thesis_catboost --algorithm catboost --shuffle --tracking-url http://10.20.20.102:8009 --storage sqlite:///tune-terakhir.db
python autogbifml --jobs 12 tune ./dataset/merged/train.parquet ./dataset/models/random_forest_tune --name thesis_random_forest --algorithm random_forest --shuffle --tracking-url http://10.20.20.102:8009 --storage sqlite:///tune-terakhir.db
python autogbifml --jobs 12 tune ./dataset/merged/train.parquet ./dataset/models/decision_tree_tune --name thesis_decision_tree --algorithm decision_tree --shuffle --tracking-url http://10.20.20.102:8009 --storage sqlite:///tune-terakhir.db
