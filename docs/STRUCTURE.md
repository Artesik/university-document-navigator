# Структура проекта

В репозитории один главный `README.md`. Документы с другими названиями содержат подробные инструкции и открываются по ссылкам из него.

```text
README.md
.gitignore
.gitattributes
FILES.sha256
app/
  app.py                   запуск Streamlit
  cli.py                   команды импорта, поиска и измерений
  INSTALL.md               установка и запуск в Windows
  modules/                 логика программы
  database/                демонстрационные инструкции JSON
  examples/                тестовые TXT, DOCX, PDF и вопросы
    EXAMPLES.md             описание учебных данных
  tests/                   модульные и интеграционные проверки
  docs/                    архитектура, руководство, методика
  scripts/                 первоначальное скачивание модели
  .streamlit/config.toml   настройки интерфейса
  setup.ps1, setup.bat     установка зависимостей
  run.bat                  запуск приложения
  requirements*.txt       зависимости
  constraints-windows-py312.txt
  model-lock.json, pyproject.toml
  CONTRIBUTING.md
docs/
  STRUCTURE.md             структура проекта
  PACKAGE_CHECK.md         проверка перед загрузкой
  validation/              скриншот, среда и реальные CSV
  ...                      журналы тестов и разработки
scripts/
  check_sources.py         проверка летнего снимка
  verify_package.py        проверка всех переданных файлов
legacy-source/
  synergy-ai-assistant.zip летние исходники без изменений
```

`app.py` и `cli.py` используют общий сервис `Navigator`. Документы читаются в `documents.py`, сохраняются через `database_manager.py`, делятся на фрагменты в `text_processing.py`. Модуль `embedding.py` вычисляет векторы E5; `index_store.py` сохраняет индекс, а `search_engine.py` выполняет поиск. Измерения находятся в `experiments.py`.

Архив летней версии нужен для подтверждения преемственности. Для запуска новой программы распаковывать его не требуется. Внутри сохранены все 28 исходных файлов с прежними именами, включая старые README. Их содержимое проверяется по SHA-256 непосредственно в ZIP.

Личные материалы практики, отчёт Word, виртуальное окружение, скачанная модель и рабочая база в комплект не входят. Файла требований и лицензии проекта нет.
