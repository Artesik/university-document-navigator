from streamlit.testing.v1 import AppTest

from modules import ui


def test_ui_import_build_search_reopen_and_empty_query(tmp_path, monkeypatch, encoder):
    monkeypatch.setattr(ui, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(ui, 'E5Encoder', lambda **kwargs: encoder)
    app = AppTest.from_file('app.py', default_timeout=30).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ['Документы', 'Поиск', 'Эксперименты']
    app.button(key='import_demo').click().run()
    assert not app.exception and any('32' in x.value for x in app.success)
    app.button(key='build_index').click().run()
    assert not app.exception and any('Индекс сохранён' in x.value for x in app.success)
    calls = len(encoder.calls)
    app.button(key='open_index').click().run()
    assert len(encoder.calls) == calls
    app.text_input(key='query').set_value('Как восстановить пароль?')
    app.button(key='search').click().run()
    assert not app.exception
    assert len([x for x in app.caption if x.value.startswith('ID фрагмента:')]) == 5
    app.text_input(key='query').set_value(' ')
    app.button(key='search').click().run()
    assert any('непустой' in x.value for x in app.warning)
