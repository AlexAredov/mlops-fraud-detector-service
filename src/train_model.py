import logging
import os

import pandas as pd
from catboost import CatBoostClassifier

from preprocessing import load_train_data, run_preproc


DROP_COLUMNS = ['name_1', 'name_2', 'street', 'post_code']
MODEL_PATH = './models/fraud_catboost.cbm'
RANDOM_STATE = 42


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info('Loading raw training data...')
    raw_train = pd.read_csv('./train_data/train.csv')
    y = raw_train['target']
    train_input = raw_train.drop(columns=DROP_COLUMNS + ['target'])

    logger.info('Preparing reference data for preprocessing...')
    reference_train = load_train_data()

    logger.info('Running preprocessing for model training...')
    x_train = run_preproc(reference_train, train_input)
    cat_features = x_train.select_dtypes(include=['object']).columns.tolist()
    logger.info('Training matrix shape: %s', x_train.shape)
    logger.info('Categorical features: %s', cat_features)

    model = CatBoostClassifier(
        iterations=150,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=25
    )
    model.fit(x_train, y, cat_features=cat_features)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    logger.info('Model saved to: %s', MODEL_PATH)


if __name__ == '__main__':
    main()
