"""Local command line entry point for reproducible checks and measurements."""
import argparse
import json
import sqlite3
from pathlib import Path

from modules.config import APP_DIR, DATA_DIR, ChunkConfig
from modules.embedding import E5Encoder
from modules.errors import NavigatorError
from modules.experiments import environment, load_labels, run_benchmark, summarize
from modules.service import Navigator


def main():
    parser = argparse.ArgumentParser(description='Локальный навигатор: документы, поиск, эксперименты')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--tokens', type=int, default=384)
    parser.add_argument('--overlap', type=int, default=64)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    sub.add_parser('import-demo')
    upload = sub.add_parser('upload')
    upload.add_argument('files', nargs='+', type=Path)
    build = sub.add_parser('build')
    build.add_argument('--batch', type=int, default=32)
    search = sub.add_parser('search')
    search.add_argument('query')
    bench = sub.add_parser('benchmark')
    bench.add_argument('--batches', type=int, nargs='+', default=[32])
    bench.add_argument('--repeats', type=int, default=3)
    bench.add_argument('--warmups', type=int, default=1)
    bench.add_argument('--labels', type=Path)
    bench.add_argument('--target-fragments', type=int)
    bench.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'doctor':
            print(json.dumps(environment(), ensure_ascii=False, indent=2))
            return 0
        nav = Navigator(args.data_dir)
        if args.command == 'import-demo':
            print('Инструкций:', nav.database.import_demo_documents(APP_DIR / 'database/knowledge_base.json'))
            return 0
        if args.command == 'upload':
            for path in args.files:
                doc = nav.upload(path.name, path.read_bytes())
                print('Добавлен:', doc['name'], doc['metadata'].get('warnings'))
            return 0
        encoder = E5Encoder(args.device, args.precision)
        config = ChunkConfig(args.tokens, args.overlap)
        if args.command == 'build':
            engine, metadata = nav.build(encoder, config, args.batch)
            print('Сохранено фрагментов:', len(engine.chunks))
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
        elif args.command == 'search':
            engine, _ = nav.open(encoder, config)
            for result in engine.search(args.query):
                print(json.dumps({'score': result.score, **result.item}, ensure_ascii=False))
        elif args.command == 'benchmark':
            rows, folder = run_benchmark(nav.database, encoder,
                args.output_dir or args.data_dir / 'experiments', batches=args.batches,
                repeats=args.repeats, warmups=args.warmups, config=config,
                labels=load_labels(args.labels) if args.labels else None,
                target_fragments=args.target_fragments, progress=print)
            print('Результаты:', folder)
            print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
            return int(any(r['status'] == 'error' for r in rows))
        return 0
    except (NavigatorError, OSError, sqlite3.Error, ValueError) as exc:
        print('Ошибка:', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
