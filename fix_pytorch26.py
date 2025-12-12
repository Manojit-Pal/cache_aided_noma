#!/usr/bin/env python3
"""
Hotfix for PyTorch 2.6 Compatibility

PyTorch 2.6 changed default weights_only=True in torch.load().
This script patches dqn_cache_final.py to add weights_only=False.

USAGE:
    python fix_pytorch26.py
    
Then re-run your comparison:
    python -m src.experiments.comparative_analysis
"""

import os
import re

file_path = 'src/caching/dqn_cache_final.py'

print("\n" + "="*70)
print("PyTorch 2.6 Compatibility Fix")
print("="*70 + "\n")

if not os.path.exists(file_path):
    print(f"❌ Error: {file_path} not found!")
    print("   Make sure you're running this from the repository root.")
    exit(1)

# Read file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already patched
if 'weights_only=False' in content:
    print("✅ File already patched!")
    print(f"   {file_path} contains weights_only=False\n")
    exit(0)

# Find and replace the torch.load() call
original_pattern = r'checkpoint = torch\.load\(filepath, map_location=self\.device\)'
replacement = 'checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)'

if re.search(original_pattern, content):
    # Apply patch
    new_content = re.sub(original_pattern, replacement, content)
    
    # Backup original
    backup_path = file_path + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Backup created: {backup_path}")
    
    # Write patched version
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Patched: {file_path}")
    print("\nChanges:")
    print("  - Added weights_only=False to torch.load()")
    print("  - Fixed PyTorch 2.6 compatibility issue")
    print("\nYou can now run:")
    print("  python -m src.experiments.comparative_analysis\n")
else:
    print("⚠️  Pattern not found in file.")
    print("   Manual fix needed:")
    print("")
    print("   Find line ~1230 in", file_path)
    print("   Change:")
    print("     checkpoint = torch.load(filepath, map_location=self.device)")
    print("   To:")
    print("     checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)")
    print("")
