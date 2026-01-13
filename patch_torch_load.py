#!/usr/bin/env python3
"""Patch torch.load calls to add weights_only=False for PyTorch 2.6+ compatibility"""
import site
import os

site_packages = site.getsitepackages()[0]

files_to_patch = [
    f"{site_packages}/speechbrain/utils/checkpoints.py",
    f"{site_packages}/pyannote/audio/core/io.py",
]

def patch_torch_load(content):
    """Patch torch.load calls handling nested parentheses properly"""
    result = []
    i = 0
    patched_count = 0

    while i < len(content):
        # Look for torch.load(
        if content[i:i+11] == 'torch.load(':
            start = i
            paren_start = i + 10  # Position of '('
            depth = 1
            j = paren_start + 1

            # Find matching closing paren
            while j < len(content) and depth > 0:
                if content[j] == '(':
                    depth += 1
                elif content[j] == ')':
                    depth -= 1
                j += 1

            # Extract the full call
            call = content[start:j]

            # Only patch if not already patched
            if 'weights_only' not in call:
                # Get the arguments (between first ( and last ))
                args = call[11:-1].rstrip()
                if args.endswith(','):
                    args = args[:-1]
                result.append(f'torch.load({args}, weights_only=False)')
                patched_count += 1
            else:
                result.append(call)
            i = j
        else:
            result.append(content[i])
            i += 1

    return ''.join(result), patched_count

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

    patched, count = patch_torch_load(content)

    with open(filepath, 'w') as f:
        f.write(patched)

    print(f"[BUILD] Patched {count} torch.load calls in: {filepath}")
