import sys
import logging

class PrintLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')
        self.logger = logging.getLogger('PrintLogger')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)

    def write(self, message):
        self.terminal.write(message)
        self.logger.info(message.strip())

    def flush(self):
        self.terminal.flush()

def set_logging(log_file):
    sys.stdout = PrintLogger(log_file)



