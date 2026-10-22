"""Russian Streamlit interface for the local university document navigator."""
import csv
import gc
import io
import sqlite3
import time

import pandas as pd
import streamlit as st
import torch

from .config import APP_DIR, BATCH_SIZES, DATA_DIR, ChunkConfig
from .embedding import E5Encoder
from .errors import NavigatorError
from .experiments import csv_bytes, load_labels, run_benchmark, summarize
from .service import Navigator

TITLE = 'Интеллектуальный навигатор по документам университета'


def show_error(exc):
    if isinstance(exc, sqlite3.Error):
        st.error(f'Не удалось выполнить операцию SQLite. Закройте другие процессы, использующие базу, '
                 f'и повторите действие. Подробности: {exc}')
    elif isinstance(exc, OSError):
        st.error(f'Ошибка доступа к локальному файлу. Проверьте свободное место и права на папку. {exc}')
    else:
        st.error(str(exc))


def get_encoder(device, precision):
    key = (device, precision)
    if st.session_state.get('encoder_key') != key:
        st.session_state.pop('encoder', None)
        st.session_state.pop('encoder_key', None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with st.spinner('Открываю локальную модель E5…'):
            encoder = E5Encoder(device=device, precision=precision)
        st.session_state['encoder'] = encoder
        st.session_state['encoder_key'] = key
    return st.session_state['encoder']


def documents_tab(nav, device, precision, config):
    st.subheader('Коллекция документов')
    st.write('Добавьте файлы или начните с демонстрационных инструкций летней практики.')
    uploads = st.file_uploader('TXT, DOCX или PDF с текстовым слоем — до 25 МБ каждый',
                               type=['txt', 'docx', 'pdf'], accept_multiple_files=True, key='uploads')
    left, right = st.columns(2)
    if left.button('Добавить файлы', key='add_files', disabled=not uploads):
        for upload in uploads:
            try:
                document = nav.upload(upload.name, upload.getvalue())
                st.success(f'{document["name"]}: текст извлечён и сохранён.')
                for warning in document['metadata'].get('warnings', []):
                    st.caption(warning)
            except (NavigatorError, OSError, sqlite3.Error) as exc:
                show_error(exc)
    if right.button('Импортировать 32 демоинструкции', key='import_demo'):
        count = nav.database.import_demo_documents(APP_DIR / 'database/knowledge_base.json')
        st.success(f'Импортировано инструкций: {count}. Повторный импорт не создаёт дубликаты.')
    docs = nav.database.get_documents()
    st.caption(f'Документов в коллекции: {len(docs)}. Учебные инструкции являются условными примерами.')
    if docs:
        st.dataframe(pd.DataFrame([{'Документ': d['name'], 'Формат': d['format'].upper(),
                                    'Страниц с текстом': str(len(d['pages'])) if d['format'] == 'pdf' else '—',
                                    'Статус': 'Текст извлечён',
                                    'Данные': 'Демонстрационные' if d['metadata']['demo'] else 'Загруженный файл'}
                                   for d in docs]), hide_index=True, width='stretch')
        with st.expander('Просмотреть текст и источник'):
            chosen = st.selectbox('Документ', [d['id'] for d in docs],
                                  format_func=lambda value: next(d['name'] for d in docs if d['id'] == value),
                                  key='document_choice')
            document = next(d for d in docs if d['id'] == chosen)
            for page in document['pages']:
                if page['page']:
                    st.caption(f'Страница PDF {page["page"]}')
                st.text(page['text'])
            if document['metadata']['demo']:
                st.write('Источник: ' + document['metadata']['source'])
                st.write('Подразделение: ' + document['metadata']['department'])
            else:
                st.download_button('Скачать сохранённый оригинал', nav.database.get_original(chosen),
                                   file_name=document['name'], key='original')
            if st.button('Удалить выбранный документ из коллекции', key='delete_document'):
                nav.database.delete_document(chosen)
                st.rerun()
    st.divider()
    st.subheader('Поисковый индекс')
    st.caption('После добавления или удаления документов перестройте индекс. Совместимость проверяется при открытии.')
    batch = st.selectbox('Размер пакета для построения', BATCH_SIZES, index=2, key='build_batch')
    build, reopen = st.columns(2)
    if build.button('Построить и сохранить индекс', type='primary', disabled=not docs, key='build_index'):
        encoder = get_encoder(device, precision)
        with st.spinner('Формирую фрагменты и строю индекс…'):
            engine, metadata = nav.build(encoder, config, batch)
        st.success(f'Индекс сохранён: {len(engine.chunks)} фрагментов. {device.upper()}, {precision.upper()}.')
        st.caption('Это рабочее построение. Для измерений с прогревом используйте «Эксперименты».')
    if reopen.button('Открыть сохранённый индекс', key='open_index'):
        engine, metadata = nav.open(get_encoder(device, precision), config)
        st.success(f'Открыт индекс: {len(engine.chunks)} фрагментов; эмбеддинги документов не пересчитывались.')
        st.caption(f'Построен: {metadata["created_at"]}; {metadata["device"].upper()}, {metadata["precision"].upper()}.')


def search_tab(nav, device, precision, config):
    st.subheader('Поиск по смыслу')
    st.write('Задайте вопрос на русском языке. Приложение покажет до пяти фрагментов и их источники.')
    with st.form('search_form'):
        query = st.text_input('Ваш вопрос', placeholder='Например: как вернуть доступ к личному кабинету?', key='query')
        submitted = st.form_submit_button('Найти фрагменты', type='primary', key='search')
    if submitted:
        if not query.strip():
            st.warning('Введите непустой поисковый запрос.')
            return
        with st.spinner('Ищу в локальных документах…'):
            encoder = get_encoder(device, precision)
            engine, metadata = nav.open(encoder, config)
            encoder.sync()
            start = time.perf_counter()
            results = engine.search(query)
            encoder.sync()
            elapsed = time.perf_counter() - start
        st.caption(f'{engine.engine_name} · Поиск: {elapsed:.3f} с. '
                   'Время открытия модели и индекса не включено.')
        st.caption(f'Эмбеддинги документов: {metadata["device"].upper()}, {metadata["precision"].upper()}; '
                   f'запрос: {device.upper()}, {precision.upper()}.')
        st.info('Косинусное сходство показывает близость текстов. Проверьте найденную выдержку по источнику.')
        for rank, result in enumerate(results, 1):
            chunk = result.item
            with st.container(border=True):
                st.markdown(f'**{rank}. {chunk["document_name"]}**')
                where = f'Страница PDF {chunk["page"]}' if chunk['page'] else f'Фрагмент {chunk["number"]}'
                st.caption(f'{where} · Сходство: {result.score:.4f}')
                st.text(chunk['text'])
                if chunk['metadata']['demo']:
                    st.caption('Учебная демонстрационная инструкция')
                    st.write('Источник: ' + chunk['metadata']['source'])
                    st.write('Подразделение: ' + chunk['metadata']['department'])
                st.caption('ID фрагмента: ' + chunk['id'])


def experiments_tab(nav, device, precision, config):
    st.subheader('Измерения построения индекса')
    st.write('Базовая конфигурация: E5, CUDA, FP32, пакет 32. Полная серия сравнений — следующий этап практики.')
    st.caption('Прогрев и загрузка модели исключены. Полное время: чтение из SQLite, фрагментация, '
               'нейросетевая обработка, сохранение индекса с записью на диск. Поиск и Hit@5 измеряются отдельно.')
    batches = st.multiselect('Пакеты для этого запуска', BATCH_SIZES, default=[32], key='experiment_batches')
    repeats = st.number_input('Измеряемые повторы', min_value=3, max_value=20, value=3, key='repeats')
    use_labels = st.checkbox('Проверить 32 размеченных вопроса к демоинструкциям', value=True, key='use_labels')
    custom_labels = st.file_uploader('Своя разметка вопросов (JSON, необязательно)', type=['json'], key='labels_file')
    if device == 'cpu':
        st.warning('Выбран CPU. Эти результаты не являются GPU-замерами.')
    if st.button('Запустить измерения', key='benchmark'):
        labels = load_labels(APP_DIR / 'examples/quality_questions.json') if use_labels else None
        if custom_labels:
            from pathlib import Path
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as folder:
                path = Path(folder) / 'labels.json'
                path.write_bytes(custom_labels.getvalue())
                labels = load_labels(path)
        encoder = get_encoder(device, precision)
        progress = st.empty()
        with st.spinner('Выполняю прогрев и измерения…'):
            rows, folder = run_benchmark(nav.database, encoder, nav.data_dir / 'experiments',
                                         batches=batches, repeats=repeats, config=config, labels=labels,
                                         progress=progress.info)
        if any(r['status'] == 'error' for r in rows):
            st.warning('В запуске есть ошибки; подробности сохранены в таблице.')
        else:
            st.success('Измерения завершены и сохранены.')
    rows = []
    for path in sorted((nav.data_dir / 'experiments').glob('*/runs.csv')):
        rows.extend(csv.DictReader(io.StringIO(path.read_text('utf-8-sig'))))
    if rows:
        summary = summarize(rows)
        st.write('Медианы успешных повторов')
        st.dataframe(pd.DataFrame(summary), hide_index=True, width='stretch')
        st.caption('Пиковая память — allocated/reserved PyTorch, включая модель. CPU: значение отсутствует. '
                   'Ускорение показывается только при наличии сопоставимого CUDA FP32/32 запуска.')
        if summary:
            chart = pd.DataFrame(summary)
            chart['Конфигурация'] = chart.apply(lambda r: f'{r.run_id[-8:]} {r.device} {r.precision} / {r.batch_size}', axis=1)
            chart = chart.set_index('Конфигурация')
            st.bar_chart(chart[['total_seconds', 'encode_seconds']], stack=False, y_label='Время, с')
            st.bar_chart(chart[['chunks_per_second']], y_label='Фрагментов/с (encode)')
            if chart['peak_allocated_mib'].notna().any():
                st.bar_chart(chart[['peak_allocated_mib', 'peak_reserved_mib']], stack=False, y_label='Память PyTorch, МиБ')
            if chart['hit_at_5'].notna().any():
                st.bar_chart(chart[['hit_at_5']], y_label='Hit@5')
        with st.expander('Все повторы и ошибки'):
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.download_button('Экспорт всех повторов CSV', csv_bytes(rows), 'experiments.csv', 'text/csv', key='export_runs')
        st.download_button('Экспорт медиан CSV', csv_bytes(summary), 'summary.csv', 'text/csv', key='export_summary')
    else:
        st.info('Сохранённых измерений пока нет.')


def main():
    st.set_page_config(page_title=TITLE, page_icon='📚', layout='wide')
    st.title(TITLE)
    st.caption('Учебный прототип · Семантический поиск · Локальная обработка документов')
    with st.sidebar:
        st.subheader('Настройки обработки')
        cuda = torch.cuda.is_available()
        device = st.selectbox('Устройство', ['cuda', 'cpu'], index=0 if cuda else 1,
                              format_func=lambda x: 'GPU (CUDA)' if x == 'cuda' else 'CPU', key='device')
        st.caption(torch.cuda.get_device_name(0) if cuda else 'CUDA в текущем PyTorch недоступна.')
        precision = st.selectbox('Точность модели', ['fp32', 'fp16'], format_func=str.upper, key='precision')
        with st.expander('Разбиение текста'):
            tokens = st.number_input('Токенов на фрагмент', min_value=32, max_value=480, value=384, key='tokens')
            overlap = st.number_input('Перекрытие токенов', min_value=0, max_value=479, value=64, key='overlap')
        st.caption('Модель: multilingual-e5-small. После первоначального скачивания интернет для поиска не нужен.')
    try:
        config = ChunkConfig(tokens, overlap)
        nav = Navigator(DATA_DIR)
        tabs = st.tabs(['Документы', 'Поиск', 'Эксперименты'])
        for tab, render in zip(tabs, (documents_tab, search_tab, experiments_tab), strict=False):
            with tab:
                try:
                    render(nav, device, precision, config)
                except (NavigatorError, OSError, sqlite3.Error, ValueError) as exc:
                    show_error(exc)
    except (NavigatorError, OSError, sqlite3.Error, ValueError) as exc:
        show_error(exc)
