#!/usr/bin/env python3
"""Patch torch.load calls to add weights_only=False for PyTorch 2.6+ compatibility"""
import re
import site
import os

site_packages = site.getsitepackages()[0]

files_to_patch = [
    f"{site_packages}/speechbrain/utils/checkpoints.py",
    f"{site_packages}/pyannote/audio/core/io.py",
]

for filepath in files_to_patch:
    if not os.path.exists(filepath):
        print(f"[BUILD] File not found (skipping): {filepath}")
        continue

    with open(filepath, 'r') as f:
        content = f.read()

    # Skip if already patched
    if 'weights_only=False' in content:
        print(f"[BUILD] Already patched: {filepath}")
        continue

    # Replace torch.load(args) with torch.load(args, weights_only=False)
    patched = re.sub(
        r'torch\.load\(([^)]+)\)',
        r'torch.load(\1, weights_only=False)',
        content
    )

    with open(filepath, 'w') as f:
        f.write(patched)

    print(f"[BUILD] Patched: {filepath}")
