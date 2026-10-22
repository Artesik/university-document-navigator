from modules.text_processing import char_ngrams, normalize_text, tokenize


def test_normalize_russian_text():
    assert normalize_text("  Ёжик   НЕ может ВОЙТИ ") == "ежик не может войти"


def test_tokenize_removes_stop_words():
    assert "пароль" in tokenize("Как мне восстановить пароль от кабинета?")
    assert "как" not in tokenize("Как мне восстановить пароль?")


def test_char_ngrams_are_created():
    assert "пар" in char_ngrams("пароль")
