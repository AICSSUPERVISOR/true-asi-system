#!/usr/bin/env python3
"""
TRUE ASI System - Advanced Reasoning Demo
==========================================

Demonstrate advanced reasoning capabilities.

Author: TRUE ASI System
Date: November 1, 2025
Version: 1.0.0
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import components
from reasoning.advanced_reasoning_engines import (
    AdvancedReasoningSystem,
    CausalRelationType
)


def demo_causal_reasoning(system):
    """Demonstrate causal reasoning"""
    print("\n" + "="*70)
    print("CAUSAL REASONING DEMONSTRATION")
    print("="*70)
    
    # Build causal graph
    system.causal.add_node("training_data", "Training Data Quality")
    system.causal.add_node("model_performance", "Model Performance")
    system.causal.add_node("user_satisfaction", "User Satisfaction")
    
    system.causal.add_edge("training_data", "model_performance", 
                          CausalRelationType.CAUSES, strength=0.9)
    system.causal.add_edge("model_performance", "user_satisfaction", 
                          CausalRelationType.CAUSES, strength=0.85)
    
    # Perform intervention
    print("\n📊 Intervention: Improving training data quality...")
    result = system.causal.intervene("training_data", "high_quality")
    print(f"   Affected nodes: {result['total_affected']}")
    
    # Find causes
    print("\n🔍 Finding causes of user satisfaction...")
    causes = system.causal.find_causes("user_satisfaction")
    for cause, strength in causes[:3]:
        print(f"   {cause}: {strength:.2f}")


def demo_probabilistic_reasoning(system):
    """Demonstrate probabilistic reasoning"""
    print("\n" + "="*70)
    print("PROBABILISTIC REASONING DEMONSTRATION")
    print("="*70)
    
    # Build Bayesian network
    system.probabilistic.add_node("weather", "Weather", ["sunny", "rainy"])
    system.probabilistic.add_node("traffic", "Traffic", ["light", "heavy"])
    system.probabilistic.add_node("arrival", "On-Time Arrival", ["yes", "no"])
    
    system.probabilistic.add_edge("weather", "traffic")
    system.probabilistic.add_edge("traffic", "arrival")
    
    # Set evidence
    print("\n📊 Evidence: Weather is rainy")
    system.probabilistic.set_evidence("weather", "rainy")
    
    # Perform inference
    print("\n🔍 Inferring probability of on-time arrival...")
    prob = system.probabilistic.infer("arrival", "yes")
    print(f"   P(on-time | rainy) = {prob:.3f}")
    
    # Find MPE
    print("\n🎯 Finding most probable explanation...")
    mpe = system.probabilistic.most_probable_explanation()
    for node, state in mpe.items():
        print(f"   {node}: {state}")


def demo_temporal_reasoning(system):
    """Demonstrate temporal reasoning"""
    print("\n" + "="*70)
    print("TEMPORAL REASONING DEMONSTRATION")
    print("="*70)
    
    # Add temporal facts
    now = datetime.now()
    system.temporal.add_fact(
        "agent_001", "status", "active",
        start_time=now - timedelta(hours=2),
        end_time=now + timedelta(hours=2)
    )
    system.temporal.add_fact(
        "agent_001", "task", "processing",
        start_time=now - timedelta(minutes=30),
        end_time=now + timedelta(minutes=30)
    )
    
    # Query at specific time
    print(f"\n📊 Querying agent status at current time...")
    facts = system.temporal.query_at_time("agent_001", "status", now)
    print(f"   Found {len(facts)} facts")
    
    # Temporal inference
    print("\n🔍 Performing temporal inference...")
    analysis = system.temporal.temporal_inference("agent_001", "status")
    print(f"   Currently true: {len(analysis.get('currently_true', []))}")
    print(f"   Past facts: {len(analysis.get('past_facts', []))}")


def demo_multi_hop_reasoning(system):
    """Demonstrate multi-hop reasoning"""
    print("\n" + "="*70)
    print("MULTI-HOP REASONING DEMONSTRATION")
    print("="*70)
    
    # Build knowledge graph
    system.multi_hop.knowledge_graph = {
        "entity_A": ["entity_B", "entity_C"],
        "entity_B": ["entity_D"],
        "entity_C": ["entity_D"],
        "entity_D": ["entity_E"]
    }
    
    # Find paths
    print("\n📊 Finding paths from entity_A to entity_E...")
    paths = system.multi_hop.find_path("entity_A", "entity_E", max_hops=4)
    print(f"   Found {len(paths)} paths:")
    for i, path in enumerate(paths[:3], 1):
        print(f"   {i}. {' → '.join(path)}")


def main():
    """Main demonstration function"""
    print("="*70)
    print("TRUE ASI SYSTEM - ADVANCED REASONING ENGINES DEMONSTRATION")
    print("="*70)
    
    # Create reasoning system
    system = AdvancedReasoningSystem()
    
    # Run demonstrations
    demo_causal_reasoning(system)
    demo_probabilistic_reasoning(system)
    demo_temporal_reasoning(system)
    demo_multi_hop_reasoning(system)
    
    # Generate and display report
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    report = system.generate_report()
    print(report)
    
    # Save report
    report_file = Path("ADVANCED_REASONING_REPORT.txt")
    report_file.write_text(report)
    print(f"\n✅ Report saved: {report_file}")
    
    print("\n" + "="*70)
    print("ADVANCED REASONING DEMONSTRATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
