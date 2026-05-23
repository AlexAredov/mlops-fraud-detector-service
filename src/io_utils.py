import json
import os

import matplotlib.pyplot as plt
import pandas as pd


DROP_COLUMNS = ['name_1', 'name_2', 'street', 'post_code']


def load_input_file(file_path):
    input_df = pd.read_csv(file_path)
    return input_df.drop(columns=DROP_COLUMNS)


def save_submission(submission, output_dir, timestamp, input_filename):
    output_filename = f"sample_submission_{timestamp}_{input_filename}"
    output_path = os.path.join(output_dir, output_filename)
    submission.to_csv(output_path, index=False)
    return output_filename


def save_feature_importances(feature_importances, output_dir, timestamp, input_filename):
    input_stem = os.path.splitext(input_filename)[0]
    output_filename = f"feature_importances_{timestamp}_{input_stem}.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(feature_importances, f, ensure_ascii=False, indent=2)

    return output_filename


def save_score_distribution(scores, output_dir, timestamp, input_filename):
    input_stem = os.path.splitext(input_filename)[0]
    output_filename = f"score_distribution_{timestamp}_{input_stem}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, density=True, color='#2f6fed', alpha=0.75)
    plt.title('Predicted Fraud Score Distribution')
    plt.xlabel('Fraud score')
    plt.ylabel('Density')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_filename
