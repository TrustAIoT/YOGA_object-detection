"""Quick end-to-end check that runs on CPU with no downloads (used by CI).

1. Builds YOGA-n/s/m/l and checks parameter counts against the paper.
2. Saves a YOGA-n checkpoint the way train.py does and reloads it with attempt_load.
3. Runs detect.py on a generated image and the PyTorch Hub entry points.

Run from the repository root:  python tests/smoke_test.py
"""
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load  # noqa: E402
from models.yolo import Model  # noqa: E402

# Parameter counts in millions (paper reports 1.9 / 7.6 / 16.3 / 33.6)
EXPECTED = {'n': 1.97, 's': 7.62, 'm': 16.37, 'l': 33.71}


def main():
    for size, expected in EXPECTED.items():
        model = Model(ROOT / f'models/YOGA-{size}.yaml', ch=3, nc=80).eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 320, 320))[0]
        assert abs(params - expected) < 0.01, f'YOGA-{size}: {params:.2f}M params, expected {expected}M'
        assert out.shape[-1] == 85, out.shape
        print(f'YOGA-{size}: {params:.2f}M params, output {tuple(out.shape)}')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = Model(ROOT / 'models/YOGA-n.yaml', ch=3, nc=80)
        model.names = [f'class{i}' for i in range(80)]
        ckpt = tmp / 'YOGA-n-random.pt'
        torch.save({'model': deepcopy(model).half(), 'ema': None, 'epoch': -1}, ckpt)

        loaded = attempt_load(str(ckpt), map_location='cpu')
        print('attempt_load: ok,', type(loaded).__name__)

        import cv2
        (tmp / 'images').mkdir()
        cv2.imwrite(str(tmp / 'images/random.jpg'), (np.random.rand(240, 320, 3) * 255).astype('uint8'))
        subprocess.run([sys.executable, str(ROOT / 'detect.py'), '--weights', str(ckpt), '--source', str(tmp / 'images'),
                        '--img', '320', '--device', 'cpu', '--project', str(tmp / 'runs')], check=True, cwd=ROOT)
        print('detect.py: ok')

        hub = torch.hub.load(str(ROOT), 'yoga_n', source='local', pretrained=False, verbose=False)
        hub(np.zeros((240, 320, 3), np.uint8))
        hub = torch.hub.load(str(ROOT), 'custom', path=str(ckpt), source='local', verbose=False)
        hub(np.zeros((240, 320, 3), np.uint8))
        print('torch.hub: ok')

    print('All smoke tests passed.')


if __name__ == '__main__':
    main()
