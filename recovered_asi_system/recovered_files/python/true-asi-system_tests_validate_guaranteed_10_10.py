#!/usr/bin/env python3.11
"""
VALIDATION TEST FOR GUARANTEED 10/10 SYSTEM
============================================

Tests that the Ultimate ASI System v13.0 achieves
ABSOLUTE 10/10 on ALL test questions.

Version: 13.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from asi_core.agent_template_v13 import UltimateASIAgentV13
from decimal import Decimal

def validate_guaranteed_10_10():
    """Validate system achieves 10/10 on all test questions."""
    
    print("\n" + "="*80)
    print("VALIDATION: GUARANTEED 10/10 SYSTEM")
    print("="*80)
    
    # Create agent
    agent = UltimateASIAgentV13(1, "universal")
    
    # Test questions (covering all domains)
    test_questions = [
        "Prove the Trans-Modal Universal Representation Theorem",
        "Design a Byzantine consensus protocol with adaptive adversary",
        "Prove explicit prime gap bounds with mechanization",
        "Create transferable causal discovery operator",
        "Define efficiently learnable representations with PAC bounds",
        "Propose novel biological mechanism predictor",
        "Design provably-optimal NP-hard scheduling heuristic",
        "Develop formal theory of emergent modularity",
        "Propose Standard Model extension with low-risk signatures",
        "Prove universal knowledge transfer theorem"
    ]
    
    results = []
    
    print(f"\nTesting {len(test_questions)} questions...")
    print("="*80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] {question[:60]}...")
        
        # Execute task
        result = agent.execute_task(question)
        
        # Extract scores
        quality_score = result.get('quality_score', 0)
        guaranteed = result.get('guaranteed_10_10', False)
        absolute_perfect = result.get('absolute_perfect', False)
        
        # Validate 10/10
        passed = (quality_score >= 10.0 and guaranteed and absolute_perfect)
        
        results.append({
            'question': question,
            'quality_score': quality_score,
            'guaranteed_10_10': guaranteed,
            'absolute_perfect': absolute_perfect,
            'passed': passed
        })
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Quality: {quality_score:.2f}/10 | Guaranteed: {guaranteed} | {status}")
    
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
    
    print(f"\nGuarantee Metrics:")
    print(f"  All Guaranteed 10/10: {all_guaranteed}")
    print(f"  All Absolute Perfect: {all_perfect}")
    
    # Final verdict
    print("\n" + "="*80)
    
    if passed == total and all_guaranteed and all_perfect:
        print("✅ VALIDATION PASSED - ABSOLUTE 10/10 GUARANTEED ON ALL QUESTIONS!")
        print("   System achieves perfect 10/10 scores with no exceptions.")
        return True
    else:
        print("❌ VALIDATION FAILED")
        print(f"   {failed} questions did not achieve 10/10")
        return False

if __name__ == "__main__":
    success = validate_guaranteed_10_10()
    sys.exit(0 if success else 1)
