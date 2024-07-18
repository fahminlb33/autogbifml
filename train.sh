#!/usr/bin/env bash

set -e

python autogbifml --jobs 12 tune ./dataset/combined/train-sel.parquet ./dataset/models/catboost_tune --name thesis_catboost_tune_v3 --algorithm catboost --shuffle --tracking-url http://10.20.20.102:8009 --storage mysql://optuna:optuna123@10.20.20.102:3306/optuna
python autogbifml --jobs 12 tune ./dataset/combined/train-sel.parquet ./dataset/models/xgboost_tune --name thesis_xgboost_tune_v3 --algorithm xgboost --shuffle --tracking-url http://10.20.20.102:8009 --storage mysql://optuna:optuna123@10.20.20.102:3306/optuna
python autogbifml --jobs 12 tune ./dataset/combined/train-sel.parquet ./dataset/models/decision_tree_tune --name thesis_decision_tree_tune_v3 --algorithm decision_tree --shuffle --tracking-url http://10.20.20.102:8009 --storage mysql://optuna:optuna123@10.20.20.102:3306/optuna
python autogbifml --jobs 12 tune ./dataset/combined/train-sel.parquet ./dataset/models/random_forest_tune --name thesis_random_forest_tune_v3 --algorithm random_forest --shuffle --tracking-url http://10.20.20.102:8009 --storage mysql://optuna:optuna123@10.20.20.102:3306/optuna

python autogbifml --jobs 12 tune ./dataset/combined_all/train-sel-10.parquet ./dataset/models_all/decision_tree_tune_v5 --name thesis_decision_tree_tune_v5 --algorithm decision_tree --shuffle --tracking-url http://10.20.20.102:8009 --storage mysql://optuna:optuna123@10.20.20.102:3306/optuna
python autogbifml --jobs 12 tune ./dataset/combined_all/train-sel-10.parquet ./dataset/models_all/xgboost_tune_v5 --name thesis_xgboost_tune_v5 --algorithm xgboost --shuffle --tracking-url http://10.20.20.102:8009 --storage mysql://optuna:optuna123@10.20.20.102:3306/optuna
python autogbifml --jobs 12 tune ./dataset/combined_all/train-sel-10.parquet ./dataset/models_all/random_forest_tune_v5 --name thesis_random_forest_tune_v5 --algorithm random_forest --shuffle --tracking-url http://10.20.20.102:8009 --storage mysql://optuna:optuna123@10.20.20.102:3306/optuna
python autogbifml --jobs 12 tune ./dataset/combined_all/train-sel-10.parquet ./dataset/models_all/catboost_tune_v5 --name thesis_catboost_tune_v5 --algorithm catboost --shuffle --tracking-url http://10.20.20.102:8009 --storage mysql://optuna:optuna123@10.20.20.102:3306/optuna

