import os
import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

sys.path.append(os.path.abspath('./src'))
from io_utils import (
    load_input_file,
    save_feature_importances,
    save_score_distribution,
    save_submission,
)
from preprocessing import load_train_data, run_preproc
from scorer import make_pred

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def wait_until_file_is_ready(file_path, checks=3, delay=1):
    previous_size = -1
    stable_checks = 0

    while stable_checks < checks:
        current_size = os.path.getsize(file_path)
        if current_size == previous_size:
            stable_checks += 1
        else:
            stable_checks = 0
            previous_size = current_size
        time.sleep(delay)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.train = load_train_data()
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            wait_until_file_is_ready(file_path)
            input_df = load_input_file(file_path)

            logger.info('Starting preprocessing')
            processed_df = run_preproc(self.train, input_df)
            
            logger.info('Making prediction')
            submission, scores, feature_importances = make_pred(processed_df, file_path)
            
            logger.info('Preparing output files')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_filename = os.path.basename(file_path)

            submission_filename = save_submission(
                submission,
                self.output_dir,
                timestamp,
                input_filename
            )
            importances_filename = save_feature_importances(
                feature_importances,
                self.output_dir,
                timestamp,
                input_filename
            )
            plot_filename = save_score_distribution(
                scores,
                self.output_dir,
                timestamp,
                input_filename
            )
            logger.info('Submission saved to: %s', submission_filename)
            logger.info('Feature importances saved to: %s', importances_filename)
            logger.info('Score distribution plot saved to: %s', plot_filename)

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()
