from typing import Any
import os
import json
import copy

from modules.core import constants


__data: dict[str, Any] = {}


def load() -> None:
    global __data

    settings_path: str = f"{constants.DATA_DIR_PATH}{constants.SETTINGS_FILENAME}"

    if os.path.exists(settings_path):
        try:
            with open(settings_path, "rt") as file:
                __data = __validate_and_fix_types(json.load(file), constants.DEFAULT_SETTINGS)
        except (json.JSONDecodeError, IOError) as exception:
            __data = copy.deepcopy(constants.DEFAULT_SETTINGS)
            print(f"Error loading settings from '{settings_path}': {exception}")
            print("Using default settings.")
    else:
        __data = copy.deepcopy(constants.DEFAULT_SETTINGS)
        print(f"Settings file '{settings_path}' not found. Using default settings.")


def save() -> None:
    settings_path: str = f"{constants.DATA_DIR_PATH}{constants.SETTINGS_FILENAME}"

    os.makedirs(constants.DATA_DIR_PATH, exist_ok=True)

    try:
        with open(settings_path, "wt") as file:
            json.dump(__data, file, indent=4)
    except IOError as exception:
        print(f"Error saving settings to '{settings_path}': {exception}")


def get_key(path: str, default_value: Any = None) -> Any:
    keys: list[str] = path.split("/")
    current: dict[str, Any] = __data

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        print(f"Malformed setting '{path}'.")
        return default_value


def set_key(path: str, value: Any) -> None:
    keys: list[str] = path.split("/")
    current: dict[str, Any] = __data

    try:
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                return
            current = current[key]

        final_key = keys[-1]
        current[final_key] = value

        save()
    except (KeyError, TypeError, IndexError):
        print(f"Failed to set setting '{path}'.")
        pass


def __validate_and_fix_types(loaded_data: dict[str, Any], default_data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, default_value in default_data.items():
        if key in loaded_data:
            loaded_value: Any = loaded_data[key]

            # If both are dictionaries, recurse.
            if isinstance(default_value, dict) and isinstance(loaded_value, dict):
                result[key] = __validate_and_fix_types(loaded_value, default_value)  # type: ignore

            # If types match, use loaded value.
            elif type(loaded_value) == type(default_value):  # type: ignore
                result[key] = loaded_value
            else:
                # Type mismatch, use default and log warning.
                print(f"Type mismatch for setting '{key}': expected {type(default_value).__name__}, "f"got {type(loaded_value).__name__}. Using default value.")  # type: ignore
                result[key] = default_value
        else:
            # Key missing in loaded data, use default
            result[key] = default_value

    return result
