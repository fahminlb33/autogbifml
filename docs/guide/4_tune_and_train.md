# Preprocess

## Tune a Model

python autogbifml --jobs -1 tune --name test_random_forest_thesis_mark1 --algorithm random_forest --trials 2 --cv 10 --shuffle --tracking-url "http://10.20.20.102:8009" ./dataset/preprocessed/america_daily/train.parquet ./dataset/preprocessed/america_daily

python autogbifml tune --name test_decision_tree_thesis_mark1 --algorithm decision_tree --trials 2 --cv 10 --shuffle --tracking-url "http://10.20.20.102:8009" ./dataset/preprocessed/america_daily/train.parquet ./dataset/preprocessed/america_daily

python autogbifml tune --name test_catboost_thesis_mark1 --algorithm catboost --trials 2 --cv 10 --shuffle --tracking-url "http://10.20.20.102:8009" ./dataset/preprocessed/america_daily/train.parquet ./dataset/preprocessed/america_daily

python autogbifml tune --name test_xgboost_thesis_mark1 --algorithm xgboost --trials 2 --cv 10 --shuffle --tracking-url "http://10.20.20.102:8009" ./dataset/preprocessed/america_daily/train.parquet ./dataset/preprocessed/america_daily

## Train a Model

## Evaluate a Model

