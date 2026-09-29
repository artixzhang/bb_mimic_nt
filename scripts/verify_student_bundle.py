#!/usr/bin/env python3
"""Verify the deployed bundle and its actual inference backend without Isaac Sim."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('bundle', type=Path)
parser.add_argument('--backend', choices=['torchscript', 'onnxruntime'], default='torchscript')
parser.add_argument('--device', default='cpu', help='TorchScript inference device.')
args = parser.parse_args()
manifest = json.loads((args.bundle / 'manifest.json').read_text())
for name, expected in manifest['artifacts_sha256'].items():
    actual = hashlib.sha256((args.bundle / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Bundle checksum mismatch: {name}')
vectors = np.load(args.bundle / 'verification_vectors.npz', allow_pickle=False)
inputs, expected = vectors['observations'], vectors['action_offsets']
if args.backend == 'torchscript':
    import torch
    model = torch.jit.load(str(args.bundle / 'student_policy.pt'), map_location=args.device).eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(inputs).to(args.device)).cpu().numpy()
else:
    import onnxruntime
    session = onnxruntime.InferenceSession(str(args.bundle / 'student_policy.onnx'), providers=['CPUExecutionProvider'])
    output = session.run(None, {'student_observation': inputs})[0]
np.testing.assert_allclose(output, expected, rtol=1.e-4, atol=1.e-5)
print(f'Bundle checksums and {args.backend} inference vectors verified; hardware validation remains separate.')
