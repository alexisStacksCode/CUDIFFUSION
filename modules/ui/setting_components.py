from typing import Any

import gradio as gr

from modules.core import shared
from modules import settings


class SettingComponent:
    __used_ids: set[int] = set({})

    def __init__(self, key: str, default_value: Any = None, **kwargs: Any) -> None:
        self.instance: gr.Component | None = None
        self.event: Any = None
        self._unique_id: int = 0

        while self._unique_id in self.__used_ids:
            self._unique_id += 1
        self.__used_ids.add(self._unique_id)

    def _add_to_shared(self) -> None:
        if self.instance is None:
            raise ReferenceError
        shared.setting_components.append(self.instance)
        shared.setting_component_values[self._unique_id] = self.instance.value

    @staticmethod
    def _on_change(unique_id: int, key: str, value: Any) -> None:
        shared.setting_component_values[unique_id] = value
        settings.set_key(key, value)


class Checkbox(SettingComponent):
    def __init__(self, key: str, default_value: Any = None, **kwargs: Any) -> None:
        super().__init__(key, default_value, **kwargs)
        kwargs["value"] = settings.get_key(key, default_value)

        self.instance = gr.Checkbox(**kwargs)
        self.event = self.instance.change(
            fn=lambda value: self._on_change(self._unique_id, key, value),  # type: ignore
            inputs=self.instance,
        )

        self._add_to_shared()


class Dropdown(SettingComponent):
    def __init__(self, key: str, default_value: Any = None, **kwargs: Any) -> None:
        super().__init__(key, default_value, **kwargs)
        kwargs["value"] = settings.get_key(key, default_value)

        self.instance = gr.Dropdown(**kwargs)
        self.event = self.instance.change(
            fn=lambda value: self._on_change(self._unique_id, key, value),  # type: ignore
            inputs=self.instance,
        )

        self._add_to_shared()
