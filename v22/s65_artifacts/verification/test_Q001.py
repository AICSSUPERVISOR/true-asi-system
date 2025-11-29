#!/usr/bin/env python3
"""
Verification test for Q001: Transfinite Fixed Point Collapse
Tests that Lean proof compiles and has no placeholders
"""

import subprocess
import sys
from pathlib import Path

def test_lean_compiles():
    """Test that Lean proof compiles successfully"""
    print("Testing Q001 Lean compilation...")
    
    proof_dir = Path(__file__).parent.parent / "proofs/lean4/Q001_TransfiniteCollapse"
    
    # Run lake build
    result = subprocess.run(
        ["lake", "build"],
        cwd=proof_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Lean compilation failed:")
        print(result.stderr)
        return False
    
    print("✅ Lean proof compiles successfully")
    return True

def test_no_placeholders():
    """Test that proof has no sorry/admit placeholders"""
    print("Testing for placeholders...")
    
    proof_file = Path(__file__).parent.parent / "proofs/lean4/Q001_TransfiniteCollapse/Basic.lean"
    
    with open(proof_file, 'r') as f:
        content = f.read()
    
    forbidden = ['sorry', 'admit', 'axiom']
    found = []
    
    for word in forbidden:
        if word in content:
            # Check it's not in a comment
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if word in line and not line.strip().startswith('--'):
                    found.append((word, i, line.strip()))
    
    if found:
        print(f"❌ Found placeholders:")
        for word, line_num, line in found:
            print(f"  Line {line_num}: {word} in '{line}'")
        return False
    
    print("✅ No placeholders found")
    return True

def test_main_theorem_present():
    """Test that main theorem is present and proven"""
    print("Testing main theorem...")
    
    proof_file = Path(__file__).parent.parent / "proofs/lean4/Q001_TransfiniteCollapse/Basic.lean"
    
    with open(proof_file, 'r') as f:
        content = f.read()
    
    # Check for main theorem
    if 'theorem collapse_iff_epsilon' not in content:
        print("❌ Main theorem 'collapse_iff_epsilon' not found")
        return False
    
    # Check it has a proof (ends with := by ... or := ...)
    if 'theorem collapse_iff_epsilon' in content and ':= by' in content:
        print("✅ Main theorem present and proven")
        return True
    
    print("❌ Main theorem not properly proven")
    return False

def main():
    """Run all tests"""
    print("="*80)
    print("Q001 VERIFICATION TESTS")
    print("="*80)
    print()
    
    tests = [
        ("Lean Compilation", test_lean_compiles),
        ("No Placeholders", test_no_placeholders),
        ("Main Theorem", test_main_theorem_present)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test '{name}' raised exception: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    print("="*80)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
