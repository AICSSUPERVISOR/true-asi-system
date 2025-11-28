"""
BRUTAL AUDIT - CYCLE 1: Code Quality
=====================================

No-holds-barred code quality audit.
100% factual results only.
"""

import os
import sys
from pathlib import Path
import ast
import re

print("=" * 80)
print("BRUTAL AUDIT - CYCLE 1: CODE QUALITY")
print("=" * 80)

# Find all Python files
python_files = list(Path('.').rglob('*.py'))
python_files = [f for f in python_files if '.git' not in str(f)]

print(f"\nTotal Python files found: {len(python_files)}")

# Analyze each file
total_lines = 0
total_functions = 0
total_classes = 0
files_with_docstrings = 0
files_with_todos = 0
files_with_placeholders = 0
placeholder_details = []

print("\nAnalyzing files...")

for py_file in python_files:
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            total_lines += len(lines)
            
            # Check for docstrings
            if '"""' in content or "'''" in content:
                files_with_docstrings += 1
            
            # Check for TODOs
            if 'TODO' in content.upper() or 'FIXME' in content.upper():
                files_with_todos += 1
            
            # Check for placeholders
            placeholder_patterns = [
                r'pass\s*#.*placeholder',
                r'NotImplementedError',
                r'raise.*not.*implemented',
                r'TODO',
                r'FIXME',
                r'XXX',
                r'HACK',
                r'placeholder',
                r'stub',
                r'mock.*implementation'
            ]
            
            for pattern in placeholder_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    files_with_placeholders += 1
                    placeholder_details.append({
                        'file': str(py_file),
                        'pattern': pattern,
                        'count': len(matches)
                    })
                    break
            
            # Parse AST
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
            except:
                pass
                
    except Exception as e:
        print(f"Error analyzing {py_file}: {e}")

print("\n" + "=" * 80)
print("CYCLE 1 RESULTS - CODE QUALITY")
print("=" * 80)

print(f"\n📊 FILE STATISTICS:")
print(f"  Total Python files: {len(python_files)}")
print(f"  Total lines of code: {total_lines:,}")
print(f"  Total functions: {total_functions}")
print(f"  Total classes: {total_classes}")

print(f"\n📝 DOCUMENTATION:")
print(f"  Files with docstrings: {files_with_docstrings}/{len(python_files)} ({files_with_docstrings/len(python_files)*100:.1f}%)")

print(f"\n⚠️  CODE QUALITY ISSUES:")
print(f"  Files with TODOs/FIXMEs: {files_with_todos}")
print(f"  Files with placeholders: {files_with_placeholders}")

if placeholder_details:
    print(f"\n❌ PLACEHOLDER DETAILS:")
    for detail in placeholder_details[:10]:  # Show first 10
        print(f"  - {detail['file']}: {detail['pattern']} ({detail['count']} occurrences)")
    if len(placeholder_details) > 10:
        print(f"  ... and {len(placeholder_details) - 10} more files")

# Check key files exist
print(f"\n🔑 KEY FILES CHECK:")
key_files = [
    'state_of_the_art_bridge.py',
    'unified_entity_layer.py',
    'perfect_orchestrator.py',
    'direct_to_s3_downloader.py',
    'models/catalog/comprehensive_hf_mappings.py',
    'master_integration.py',
    'unified_interface.py'
]

for key_file in key_files:
    exists = Path(key_file).exists()
    status = "✅" if exists else "❌"
    print(f"  {status} {key_file}")
    
    if exists:
        # Check file size
        size = Path(key_file).stat().st_size
        print(f"      Size: {size:,} bytes")

# Calculate quality score
quality_score = 100
if files_with_placeholders > 0:
    quality_score -= (files_with_placeholders / len(python_files)) * 50
if files_with_todos > 0:
    quality_score -= (files_with_todos / len(python_files)) * 20
if files_with_docstrings / len(python_files) < 0.5:
    quality_score -= 20

print(f"\n📈 QUALITY SCORE: {quality_score:.1f}/100")

print("\n" + "=" * 80)
print("CYCLE 1 COMPLETE")
print("=" * 80)

# Save results
with open('audit_cycle1_results.txt', 'w') as f:
    f.write(f"CYCLE 1 RESULTS\n")
    f.write(f"===============\n\n")
    f.write(f"Python files: {len(python_files)}\n")
    f.write(f"Lines of code: {total_lines:,}\n")
    f.write(f"Functions: {total_functions}\n")
    f.write(f"Classes: {total_classes}\n")
    f.write(f"Files with docstrings: {files_with_docstrings}\n")
    f.write(f"Files with TODOs: {files_with_todos}\n")
    f.write(f"Files with placeholders: {files_with_placeholders}\n")
    f.write(f"Quality score: {quality_score:.1f}/100\n")

print("\n✅ Results saved to audit_cycle1_results.txt")
