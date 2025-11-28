"""
BRUTAL AUDIT - CYCLE 2: Integration
====================================

Test all imports, dependencies, and connections.
100% factual results only.
"""

import sys
import importlib
from pathlib import Path

print("=" * 80)
print("BRUTAL AUDIT - CYCLE 2: INTEGRATION")
print("=" * 80)

# Key modules to test
key_modules = [
    'state_of_the_art_bridge',
    'unified_entity_layer',
    'perfect_orchestrator',
    'direct_to_s3_downloader',
    'master_integration',
    'unified_interface',
]

print("\n🔍 TESTING KEY MODULE IMPORTS...")

import_results = {}
for module_name in key_modules:
    try:
        module = importlib.import_module(module_name)
        import_results[module_name] = {
            'status': 'SUCCESS',
            'error': None,
            'has_main': hasattr(module, '__all__') or hasattr(module, 'main')
        }
        print(f"  ✅ {module_name}")
    except Exception as e:
        import_results[module_name] = {
            'status': 'FAILED',
            'error': str(e),
            'has_main': False
        }
        print(f"  ❌ {module_name}: {str(e)[:100]}")

# Test specific integrations
print("\n🔗 TESTING SPECIFIC INTEGRATIONS...")

# Test 1: State-of-the-art bridge
print("\n1. State-of-the-Art Bridge:")
try:
    from state_of_the_art_bridge import get_bridge, get_status, ModelCapability
    bridge = get_bridge()
    status = get_status()
    print(f"  ✅ Bridge initialized")
    print(f"  ✅ Total models in registry: {status.get('total_models', 0)}")
    print(f"  ✅ ModelCapability enum: {len([c for c in ModelCapability])}")
except Exception as e:
    print(f"  ❌ Failed: {str(e)[:200]}")

# Test 2: Comprehensive HF mappings
print("\n2. Comprehensive HF Mappings:")
try:
    sys.path.insert(0, 'models/catalog')
    from comprehensive_hf_mappings import COMPREHENSIVE_HF_MAPPINGS, get_all_models
    models = get_all_models()
    print(f"  ✅ Mappings loaded")
    print(f"  ✅ Total models mapped: {len(COMPREHENSIVE_HF_MAPPINGS)}")
    print(f"  ✅ get_all_models() returns: {len(models)}")
except Exception as e:
    print(f"  ❌ Failed: {str(e)[:200]}")

# Test 3: Unified Entity Layer
print("\n3. Unified Entity Layer:")
try:
    from unified_entity_layer import UnifiedEntityLayer, TaskType
    entity = UnifiedEntityLayer()
    print(f"  ✅ Entity layer initialized")
    print(f"  ✅ TaskType enum exists")
except Exception as e:
    print(f"  ❌ Failed: {str(e)[:200]}")

# Test 4: Perfect Orchestrator
print("\n4. Perfect Orchestrator:")
try:
    from perfect_orchestrator import PerfectOrchestrator, OrchestrationMode
    orchestrator = PerfectOrchestrator()
    print(f"  ✅ Orchestrator initialized")
    print(f"  ✅ OrchestrationMode enum exists")
except Exception as e:
    print(f"  ❌ Failed: {str(e)[:200]}")

# Test 5: S3 connectivity
print("\n5. AWS S3 Connectivity:")
try:
    import boto3
    import os
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )
    bucket = 'asi-knowledge-base-898982995956'
    response = s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
    print(f"  ✅ S3 connection successful")
    print(f"  ✅ Bucket accessible: {bucket}")
except Exception as e:
    print(f"  ❌ Failed: {str(e)[:200]}")

# Test 6: Cross-module integration
print("\n6. Cross-Module Integration:")
try:
    from state_of_the_art_bridge import StateOfTheArtBridge
    bridge = StateOfTheArtBridge()
    # Check if bridge has entity_layer and orchestrator
    has_entity = hasattr(bridge, 'entity_layer')
    has_orchestrator = hasattr(bridge, 'orchestrator')
    has_registry = hasattr(bridge, 'model_registry')
    
    print(f"  {'✅' if has_entity else '❌'} Bridge has entity_layer: {has_entity}")
    print(f"  {'✅' if has_orchestrator else '❌'} Bridge has orchestrator: {has_orchestrator}")
    print(f"  {'✅' if has_registry else '❌'} Bridge has model_registry: {has_registry}")
    
    if has_registry:
        print(f"  ✅ Registry size: {len(bridge.model_registry)}")
except Exception as e:
    print(f"  ❌ Failed: {str(e)[:200]}")

# Calculate integration score
print("\n" + "=" * 80)
print("CYCLE 2 RESULTS - INTEGRATION")
print("=" * 80)

successful_imports = sum(1 for r in import_results.values() if r['status'] == 'SUCCESS')
total_imports = len(import_results)

print(f"\n📊 IMPORT SUCCESS RATE: {successful_imports}/{total_imports} ({successful_imports/total_imports*100:.1f}%)")

print(f"\n📋 DETAILED RESULTS:")
for module, result in import_results.items():
    status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
    print(f"  {status_icon} {module}: {result['status']}")
    if result['error']:
        print(f"      Error: {result['error'][:100]}")

integration_score = (successful_imports / total_imports) * 100

print(f"\n📈 INTEGRATION SCORE: {integration_score:.1f}/100")

print("\n" + "=" * 80)
print("CYCLE 2 COMPLETE")
print("=" * 80)

# Save results
with open('audit_cycle2_results.txt', 'w') as f:
    f.write(f"CYCLE 2 RESULTS\n")
    f.write(f"===============\n\n")
    f.write(f"Successful imports: {successful_imports}/{total_imports}\n")
    f.write(f"Integration score: {integration_score:.1f}/100\n\n")
    f.write(f"Details:\n")
    for module, result in import_results.items():
        f.write(f"  {module}: {result['status']}\n")
        if result['error']:
            f.write(f"    Error: {result['error']}\n")

print("\n✅ Results saved to audit_cycle2_results.txt")
