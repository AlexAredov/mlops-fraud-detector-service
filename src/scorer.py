import pandas as pd
import logging
from catboost import CatBoostClassifier

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier()
model.load_model('./models/my_catboost.cbm')

# Define optimal threshold
model_th = 0.98
logger.info('Pretrained model imported successfully...')


def get_top_feature_importances(feature_names, top_n=5):
    importances = model.get_feature_importance()
    feature_importances = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return {feature: float(importance) for feature, importance in feature_importances}


# Make prediction
def make_pred(dt, path_to_file):
    scores = model.predict_proba(dt)[:, 1]

    # Make submission dataframe
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': (scores > model_th) * 1
    })
    feature_importances = get_top_feature_importances(dt.columns)
    logger.info('Prediction complete for file: %s', path_to_file)

    return submission, scores, feature_importances
