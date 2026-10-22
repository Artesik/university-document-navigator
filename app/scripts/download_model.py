"""One-time public model download. Sends no documents or queries."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
MODEL_ID = 'intfloat/multilingual-e5-small'


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main():
    lock_path = APP / 'model-lock.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text('utf-8'))
        info = get_json(f'https://huggingface.co/api/models/{MODEL_ID}/revision/{lock["revision"]}')
    else:
        info = get_json(f'https://huggingface.co/api/models/{MODEL_ID}')
        lock = {'model_id': MODEL_ID, 'revision': info['sha']}
        lock_path.write_text(json.dumps(lock, indent=2), encoding='utf-8')
    target = APP / 'models/multilingual-e5-small'
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for item in info['siblings']:
        name = item['rfilename']
        if name.startswith(('onnx/', 'openvino/')) or not name.endswith(('.json', '.safetensors', '.model')):
            continue
        output = target / name
        output.parent.mkdir(parents=True, exist_ok=True)
        # Validate existing files against the previously completed manifest.
        manifest = target / 'model-manifest.json'
        old = json.loads(manifest.read_text('utf-8')) if manifest.exists() else {}
        if old.get('revision') == lock['revision'] and output.exists():
            if hashlib.sha256(output.read_bytes()).hexdigest() == old.get('files', {}).get(name):
                files[name] = old['files'][name]
                continue
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{lock["revision"]}/{name}'
        print('Downloading', name, flush=True)
        temporary = output.with_name(output.name + '.part')
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        files[name] = hashlib.sha256(output.read_bytes()).hexdigest()
    if 'model.safetensors' not in files or 'modules.json' not in files:
        raise RuntimeError('Incomplete model repository')
    (target / 'model-manifest.json').write_text(json.dumps({**lock, 'files': files}, indent=2), encoding='utf-8')
    print('Model saved locally:', lock['revision'], flush=True)


if __name__ == '__main__':
    main()
