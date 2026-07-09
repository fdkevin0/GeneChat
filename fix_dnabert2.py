#!/usr/bin/env python3
"""One-time patch: fix DNABERT-2 ALiBi compatibility with PyTorch 2.12.
Run once before training: .venv/bin/python fix_dnabert2.py
"""
import os, glob

cache = os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/"
    "zhihan1996/DNABERT_hyphen_2_hyphen_117M"
)
files = glob.glob(f"{cache}/*/bert_layers.py")
if not files:
    print("ERROR: DNABERT-2 cache not found. Run training once first to download it.")
    exit(1)

target = files[0]
print(f"Patching: {target}")

with open(target) as f:
    content = f.read()

old = """def rebuild_alibi_tensor(self,
                             size: int,
                             device: Optional[Union[torch.device, str]] = None):
        # Alibi"""

new = """def rebuild_alibi_tensor(self,
                             size: int,
                             device: Optional[Union[torch.device, str]] = None):
        # Alibi
        if device is None:
            device = torch.device("cpu")"""

if "if device is None:" in content and "device = torch.device" in content:
    print("Already patched — nothing to do")
    exit(0)

if old not in content:
    print("ERROR: Could not find the target code block. Already patched?")
    exit(1)

content = content.replace(old, new)

with open(target, "w") as f:
    f.write(content)

print("✅ DNABERT-2 patched successfully")
