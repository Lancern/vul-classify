from typing import *

import json


class ConfigurationManager:
    def __init__(self, file_name: str):
        with open(file_name, 'r') as fp:
            self._entries = json.load(fp)

    def get(self, name: str, default: Any = None) -> Any:
        return self._entries.get(name, default)


global_config: Optional[ConfigurationManager] = None


def init_config(file_name: str) -> None:
    global global_config
    global_config = ConfigurationManager(file_name)


def app_config() -> ConfigurationManager:
    return global_config
