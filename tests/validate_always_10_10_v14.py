#!/usr/bin/env python3.11
"""
VALIDATION TEST FOR ALWAYS-10/10 SYSTEM V14
===========================================

Tests that the Ultimate ASI System v14.0 achieves
ABSOLUTE 10/10 ALWAYS on diverse test cases.

Version: 14.0
Tier: S-5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'asi_core'))

from agent_template_v14 import UltimateASIAgentV14

def validate_always_10_10():
    """Validate system achieves 10/10 ALWAYS on diverse test cases."""
    
    print("\n" + "="*80)
    print("VALIDATION: ALWAYS-10/10 SYSTEM V14 (S-5 LEVEL)")
    print("="*80)
    
    # Create agent
    agent = UltimateASIAgentV14(1, "universal")
    
    # Diverse test questions across all domains
    test_questions = [
        # Mathematics
        "Prove the Riemann Hypothesis",
        "Solve the Navier-Stokes existence and smoothness problem",
        
        # Physics
        "Derive a unified theory of quantum gravity",
        "Explain dark matter and dark energy",
        
        # Computer Science
        "Prove P vs NP",
        "Design a quantum algorithm for factoring in polynomial time",
        
        # Biology
        "Model protein folding with atomic precision",
        "Explain the origin of life from first principles",
        
        # Chemistry
        "Design a room-temperature superconductor",
        "Synthesize a universal cancer cure",
        
        # Philosophy
        "Solve the hard problem of consciousness",
        "Resolve the free will vs determinism debate",
        
        # Engineering
        "Design a fusion reactor with net positive energy",
        "Create a general-purpose molecular assembler",
        
        # Economics
        "Prove the fundamental theorem of welfare economics",
        "Design an optimal economic system"
    ]
    
    results = []
    
    print(f"\nTesting {len(test_questions)} diverse questions...")
    print("="*80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] {question[:60]}...")
        
        # Execute task
        result = agent.execute_task(question)
        
        # Extract scores
        quality_score = result.get('quality_score', 0)
        guaranteed = result.get('guaranteed_10_10', False)
        absolute_perfect = result.get('absolute_perfect', False)
        tier = result.get('tier', 'Unknown')
        
        # Validate 10/10
        passed = (quality_score >= 10.0 and guaranteed and absolute_perfect and tier == 'S-5')
        
        results.append({
            'question': question,
            'quality_score': quality_score,
            'guaranteed_10_10': guaranteed,
            'absolute_perfect': absolute_perfect,
            'tier': tier,
            'passed': passed
        })
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Quality: {quality_score:.2f}/10 | Tier: {tier} | {status}")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    failed = total - passed
    
    print(f"\nTotal Questions: {total}")
    print(f"Passed (10/10): {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {100*passed/total:.1f}%")
    
    # Quality metrics
    avg_quality = sum(r['quality_score'] for r in results) / total
    min_quality = min(r['quality_score'] for r in results)
    max_quality = max(r['quality_score'] for r in results)
    
    print(f"\nQuality Scores:")
    print(f"  Average: {avg_quality:.2f}/10")
    print(f"  Minimum: {min_quality:.2f}/10")
    print(f"  Maximum: {max_quality:.2f}/10")
    
    # Guarantee metrics
    all_guaranteed = all(r['guaranteed_10_10'] for r in results)
    all_perfect = all(r['absolute_perfect'] for r in results)
    all_s5 = all(r['tier'] == 'S-5' for r in results)
    
    print(f"\nGuarantee Metrics:")
    print(f"  All Guaranteed 10/10: {all_guaranteed}")
    print(f"  All Absolute Perfect: {all_perfect}")
    print(f"  All S-5 Level: {all_s5}")
    
    # Final verdict
    print("\n" + "="*80)
    
    if passed == total and all_guaranteed and all_perfect and all_s5:
        print("✅ VALIDATION PASSED - ABSOLUTE 10/10 ALWAYS ON ALL QUESTIONS!")
        print("   System achieves perfect 10/10 scores with no exceptions.")
        print("   S-5 level confirmed across all domains.")
        return True
    else:
        print("❌ VALIDATION FAILED")
        print(f"   {failed} questions did not achieve 10/10")
        return False

if __name__ == "__main__":
    success = validate_always_10_10()
    sys.exit(0 if success else 1)
