import logging
import os
from typing import Callable, Optional, Any
from datetime import datetime
from functools import wraps
from transformers.utils import logging as t_logging


LOGGER_NAME = "synthcoder_logger"
FORMAT = '%(asctime)s %(name)s %(funcName)s %(levelno)s %(lineno)d %(message)s'

current_datetime= datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
logger = logging.getLogger(LOGGER_NAME)
t_logging.set_verbosity_warning()


def setup_logger() -> None:
    """
    Configure logging settings and ensure the log directory exists.
    Import within this function allows to avoid the circular import issue. 

    Returns:
    ========
    None
    """
    from synthcoder_project.synthcoder_config import ACTION_LOGGER_DIR, ACTION_LOGGER_LEVEL
    os.makedirs(ACTION_LOGGER_DIR, exist_ok=True)
    logging.basicConfig(filename=os.path.join(ACTION_LOGGER_DIR, f"{current_datetime}.log"), level=ACTION_LOGGER_LEVEL, format=FORMAT)


def create_logger(module_name: str) -> logging.Logger:
    """
    Create a logger with the specified module name.

    Parameters:
    ===========
    module_name: Str. The name used to identify the logger.

    Returns:
    ========
    logging.Logger. A logger object for logging messages in the specified module.
    """
    return logging.getLogger(f"{module_name}")


def logged(level: int=logging.DEBUG, message: str=None, name: str=None) -> Callable:
    """
    Decorator that logs a message each time the decorated function is called.
    The message is logged at the specified logging level and logger name.

    Parameters:
    ===========
    level: Int. <Optional> The logging level, defaulting to `logging.DEBUG`. 
        Determines the severity level of the log.     
    message: Str. <Optional> The message to be logged when the decorated function is called.
        If not provided, defaults to no message.       
    name: Str. <Optional> The logger's name. If not specified, defaults to the module
        name where the decorated function is defined.

    Returns:
    ========
    Function. The original function wrapped with logging capabilities.
    """
    def decorate(func):
        try:
            logname = name if name else func.__module__  # determine the logger name, using the specified name or defaulting to the module name
        except AttributeError:
            logname = "_"
        try:
            log_message = message.format(cls=func.__qualname__.split('.')[0], func=func.__name__) if message else f"Running {func.__name__}"
        except AttributeError:
            log_message = "NA - Could not log message!!! - logger could not access attribute .__qualname__ or .__name__"

        logger = logging.getLogger(logname)

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(level, log_message)
            return func(*args, **kwargs)
        return wrapper
    return decorate
