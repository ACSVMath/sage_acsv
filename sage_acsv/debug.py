"""Utilities for logging and debugging."""

import time
import logging

logging.basicConfig()
acsv_logger = logging.getLogger("sage_acsv")


class Timer:
    def __init__(self):
        self._time_created = time.monotonic()
        self._last_checkpoint = self._time_created

    def checkpoint(self, name=None):
        if name:
            time_diff = time.monotonic() - self._last_checkpoint
            acsv_logger.info(
                f"{bcolors.OKBLUE}Timer:{bcolors.ENDC} "
                f"Executed {name} in {time_diff} seconds."
            )

        self._last_checkpoint = time.monotonic()

    def total(self, name):
        if self.show_time and name:
            time_diff = time.monotonic() - self._time_created
            acsv_logger.info(
                f"{bcolors.OKBLUE}Timer:{bcolors.ENDC}Total runtime "
                f"of {name} is {time_diff} seconds."
            )


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
