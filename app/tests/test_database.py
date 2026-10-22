from modules.database_manager import DatabaseManager


def test_database_initialization_and_history(tmp_path):
    database = DatabaseManager(tmp_path / "assistant.db", "database/knowledge_base.json")
    database.initialize()
    assert len(database.get_knowledge_items()) == 32
    history_id = database.add_history("Как восстановить пароль?", 1, 0.92)
    database.add_feedback(history_id, True, "Полезно")
    history = database.get_history()
    assert history[0]["query"] == "Как восстановить пароль?"
    database.clear_history()
    assert database.get_history() == []
