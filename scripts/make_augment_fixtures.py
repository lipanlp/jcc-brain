"""从人工真值选定的截图经CoreGraphics生成小图夹具；不生成或改写真值。"""
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jccbrain.vision.eye.augments import AugmentCatalog  # noqa: E402
from jccbrain.vision.eye.capture import load_image  # noqa: E402


def save(path, arrays):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            info = zipfile.ZipInfo(key + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())


def main():
    samples = ROOT / 'var/vision-module-optimization/replay-samples'
    old = Path('game-eye/samples')
    catalog = AugmentCatalog.from_db(SimpleNamespace(path=Path('/tmp/jcc-vf-db.sqlite3'), season_id='18_S19'))
    arrays, entries, cases, official = {}, [], [], []
    for row in catalog.loaded:
        official.append(cv2.resize(load_image(catalog.file(row)), (16, 16), interpolation=cv2.INTER_AREA))
        entries.append(dict(row, index=len(official)-1))
    arrays['official'] = np.stack(official)
    truth = {r['_file']: r for r in map(json.loads, (ROOT / 'tests/fixtures/augment_truth.jsonl').read_text().splitlines())}
    chosen = ['augment-taken/0003.png', 'augment-taken/0005.png', 'augment-taken/0007.png',
              'augment-pick/0001.png', '0000.jpg', '0051.jpg', '0063.jpg', 'spectate/0003.png']
    for key in chosen:
        image = load_image((samples if '/' in key else old) / key)
        arrays[key] = image[280:340, 450:610]
        cases.append(dict(file=key, array=key, rect=[450, 280, 610, 340],
                          expected=truth.get(key, {}).get('augments_taken', []), spectate=key.startswith('spectate/')))
        if key.startswith('spectate/'):
            arrays['spectate_strip'] = image[102:784, 1708:1743]
    fixtures = ROOT / 'tests/fixtures'
    save(fixtures / 'augment_cells.npz', arrays)
    (fixtures / 'augment_cells.json').write_text(json.dumps(dict(entries=entries, cases=cases), ensure_ascii=False, indent=2)+'\n')
    eye = load_image(samples / 'spectate/0003.png')[135:154, 1710:1741]
    save(ROOT / 'src/jccbrain/vision/eye/augment_ui.npz', dict(spectate_eye=eye))
    size = (fixtures / 'augment_cells.npz').stat().st_size
    assert size <= 300_000, size
    print(f'{len(entries)} official icons, {len(cases)} crops; {size} bytes')


if __name__ == '__main__':
    main()
