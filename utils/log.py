import sys
import logging


class Log():
    """Logging class for reusability in all modules"""
    def __init__(self, log_id):
        ##### Logging #####
        self.log = logging.getLogger(__name__)
        logging.basicConfig(filename=f"./logs/log_{log_id}.log",
                            filemode='a',
                            format="%(asctime)s.%(msecs)03d - %(filename)s - %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.DEBUG)

        def handle_exception(exc_type, exc_value, exc_traceback):
            self.log.error("Uncaught exception", exc_info=(
                exc_type, exc_value, exc_traceback))

        sys.excepthook = handle_exception
