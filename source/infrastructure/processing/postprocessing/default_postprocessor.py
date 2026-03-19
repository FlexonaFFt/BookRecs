from __future__ import annotations

from typing import Any

from source.application.ports import PostProcessorPort
from source.domain.entities import FinalItem


# Применяет финальные правила очистки рекомендаций.
class DefaultPostprocessor(PostProcessorPort):

    def apply(
        self,
        items: list[FinalItem],
        seen_items: set[Any],
        top_k: int,
    ) -> list[FinalItem]:
        if top_k <= 0:
            return []

        out: list[FinalItem] = []
        used: set[Any] = set()
        for item in items:
            if item.item_id in seen_items:
                continue
            if item.item_id in used:
                continue
            used.add(item.item_id)
            out.append(item)
            if len(out) >= top_k:
                break
        return out
