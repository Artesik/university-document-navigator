"""Feedback service kept separate from the UI for testability."""

from __future__ import annotations

from .database_manager import DatabaseManager


class FeedbackManager:
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    def submit(self, history_id: int | None, helpful: bool, comment: str = "") -> bool:
        if history_id is None:
            return False
        self.database.add_feedback(history_id, helpful, comment)
        return True
