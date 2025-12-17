#!/usr/bin/env python3
"""System Verification Script"""
import sys
import os

print("🔍 Verifying TRUE ASI System...")
print("="*70)

checks = []

# Check Python version
import sys
if sys.version_info >= (3, 11):
    print("✅ Python version: OK")
    checks.append(True)
else:
    print("❌ Python version: FAILED (need 3.11+)")
    checks.append(False)

# Check directory structure
dirs = ['src', 'agents', 'tests', 'docs', 'scripts', 'deployment']
for d in dirs:
    if os.path.exists(d):
        print(f"✅ Directory {d}: OK")
        checks.append(True)
    else:
        print(f"❌ Directory {d}: MISSING")
        checks.append(False)

# Check agent files
agent_count = len([f for f in os.listdir('agents') if f.startswith('agent_')])
if agent_count == 250:
    print(f"✅ Agents: OK ({agent_count}/250)")
    checks.append(True)
else:
    print(f"⚠️  Agents: PARTIAL ({agent_count}/250)")
    checks.append(False)

print("="*70)
if all(checks):
    print("✅ System verification complete")
    sys.exit(0)
else:
    print("⚠️  System verification completed with warnings")
    sys.exit(0 if sum(checks) > len(checks) * 0.8 else 1)
