from abc import ABC
from midst import LOGS_DIR
from typing import Any, Iterable

import os
import pickle


class Logger(ABC):
    """
    A class for logging results, predictions and other metrics of interest throughout experiments.
    """

    def __init__(self, log_dir: str = LOGS_DIR, experiment_name: str = None, max_elements: int = 10000):
        """
        Constructor for the Logger

        :param log_dir: (str) Path to the general log directory in which to save results.
        :param experiment_name: (str) Name of the experiment to log. Logs will be saved under:
         .../log_dir/experiment_name.
         :param log_dir: Maximal number of elements allowed to be saved between each flush, per-variable.
        """

        # Setup
        self._max_elements = max_elements
        self._save_dir = os.path.join(log_dir, experiment_name)
        self.save_dir_train = os.path.join(self._save_dir, "Train")
        self.save_dir_val = os.path.join(self._save_dir, "Val")
        self.save_dir_test = os.path.join(self._save_dir, "Test")
        os.makedirs(self._save_dir, exist_ok=True)
        os.makedirs(self.save_dir_train, exist_ok=True)
        os.makedirs(self.save_dir_val, exist_ok=True)
        os.makedirs(self.save_dir_test, exist_ok=True)

        self._vars_to_log = {}
        self._vars_to_log_counters = {}

    def log_variable(self, var: Any, var_name: str, ignore_cap: bool = False):
        """
        Method for collecting variables to log

        :param var: (Any) The variable to log
        :param var_name: (str) The name of the variable to be logged
        :param ignore_cap: (str) Whether to ignore the maximum cap for this specific variable

        :return: None
        """

        if var_name in self._vars_to_log:
            if (self._vars_to_log_counters[var_name] >= self._max_elements) and not ignore_cap:
                self._vars_to_log[var_name][self._vars_to_log_counters[var_name] % self._max_elements] = var

            else:
                self._vars_to_log[var_name].append(var)

            self._vars_to_log_counters[var_name] += 1

        else:
            self._vars_to_log[var_name] = [var]
            self._vars_to_log_counters[var_name] = 1

    def flush(self, variables: Iterable[str] = None, save_dir: str = None):
        """
        A method for writing all the currently collected logs and clearing the memory taken for holding them

        :param variables: (List[str]) List of variables to flush. If None flushes all currently logged variables.
        Defaults to None.
        :param save_dir: (str) The location in which to save the logs. If None then saves to self._save_dir.
        Defaults to None.

        :return: None
        """

        save_dir = self._save_dir if save_dir is None else save_dir

        for var_name in self._vars_to_log:
            if var_name in variables:
                file_path = os.path.join(
                    save_dir,
                    f"{var_name}_{self._vars_to_log_counters[var_name]}.pkl"
                )

                with open(file_path, 'wb') as pickle_file:
                    pickle.dump(file=pickle_file, obj=self._vars_to_log[var_name])

    def clear_logs(self):
        self._vars_to_log = {}
        self._vars_to_log_counters = {}

    @property
    def save_dir(self):
        """
        Class property, returns the directory in which all logs are saved.

        :return: (str) Path the to directory
        """

        return self._save_dir

    @property
    def logged_vars(self) -> Iterable:
        """
        Class property, returns the names of all of the currently logged variables

        :return: (str) Path the to directory
        """

        return self._vars_to_log.keys()
