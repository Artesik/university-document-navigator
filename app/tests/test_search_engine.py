import json
from pathlib import Path

from modules.search_engine import SemanticSearchEngine

ITEMS = json.loads(Path("database/knowledge_base.json").read_text(encoding="utf-8"))["items"]


def engine():
    return SemanticSearchEngine(ITEMS, use_transformer=False)


def test_password_query_finds_account_answer():
    result = engine().search("Я забыл данные и теперь не могу войти в аккаунт")
    assert result.item is not None
    assert result.item["id"] == 1


def test_lms_query_finds_material_upload():
    result = engine().search("Хочу добавить презентацию и лекцию в курс LMS")
    assert result.item is not None
    assert result.item["id"] == 4


def test_category_filter_is_respected():
    result = engine().search("Нужно оформить заявление", category="Кадровые вопросы")
    assert result.item is not None
    assert result.item["category"] == "Кадровые вопросы"


def test_unknown_query_is_rejected():
    result = engine().search("квантовый телепортатор на марсе")
    assert result.item is None
