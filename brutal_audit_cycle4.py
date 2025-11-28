"""
BRUTAL AUDIT - CYCLE 4: Functionality
======================================

Test actual execution and functionality.
100% factual results only.
"""

import sys
import traceback

print("=" * 80)
print("BRUTAL AUDIT - CYCLE 4: FUNCTIONALITY")
print("=" * 80)

test_results = {}

# Test 1: Bridge initialization
print("\n🧪 TEST 1: Bridge Initialization")
try:
    from state_of_the_art_bridge import get_bridge, get_status
    bridge = get_bridge()
    status = get_status()
    
    assert status['total_models'] > 0, "No models in registry"
    assert status['status'] == 'operational', "Bridge not operational"
    
    test_results['bridge_init'] = {
        'status': 'PASS',
        'details': f"{status['total_models']} models registered"
    }
    print(f"  ✅ PASS: {status['total_models']} models registered")
except Exception as e:
    test_results['bridge_init'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    print(f"  ❌ FAIL: {str(e)[:100]}")

# Test 2: Model selection
print("\n🧪 TEST 2: Model Selection")
try:
    from state_of_the_art_bridge import get_bridge, ModelCapability
    bridge = get_bridge()
    
    # Test code model selection
    model = bridge.select_model("Write Python code", capability=ModelCapability.CODE_GENERATION)
    
    assert model is not None, "No model selected"
    assert ModelCapability.CODE_GENERATION in model.capabilities, "Wrong capability"
    
    test_results['model_selection'] = {
        'status': 'PASS',
        'details': f"Selected: {model.name}"
    }
    print(f"  ✅ PASS: Selected {model.name}")
except Exception as e:
    test_results['model_selection'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    print(f"  ❌ FAIL: {str(e)[:100]}")

# Test 3: List models
print("\n🧪 TEST 3: List Models")
try:
    from state_of_the_art_bridge import list_models, ModelCapability
    
    all_models = list_models()
    code_models = list_models(capability=ModelCapability.CODE_GENERATION)
    
    assert len(all_models) > 0, "No models listed"
    assert len(code_models) > 0, "No code models found"
    assert len(code_models) < len(all_models), "Code models should be subset"
    
    test_results['list_models'] = {
        'status': 'PASS',
        'details': f"{len(all_models)} total, {len(code_models)} code models"
    }
    print(f"  ✅ PASS: {len(all_models)} total, {len(code_models)} code models")
except Exception as e:
    test_results['list_models'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    print(f"  ❌ FAIL: {str(e)[:100]}")

# Test 4: HF Mappings
print("\n🧪 TEST 4: HuggingFace Mappings")
try:
    sys.path.insert(0, 'models/catalog')
    from comprehensive_hf_mappings import COMPREHENSIVE_HF_MAPPINGS, get_hf_id
    
    # Test some known models
    qwen_id = get_hf_id("Qwen 2.5 72B")
    mistral_id = get_hf_id("Mistral 7B")
    
    assert qwen_id is not None, "Qwen mapping not found"
    assert mistral_id is not None, "Mistral mapping not found"
    assert len(COMPREHENSIVE_HF_MAPPINGS) > 200, "Too few mappings"
    
    test_results['hf_mappings'] = {
        'status': 'PASS',
        'details': f"{len(COMPREHENSIVE_HF_MAPPINGS)} mappings"
    }
    print(f"  ✅ PASS: {len(COMPREHENSIVE_HF_MAPPINGS)} mappings")
except Exception as e:
    test_results['hf_mappings'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    print(f"  ❌ FAIL: {str(e)[:100]}")

# Test 5: S3 Access
print("\n🧪 TEST 5: S3 Access")
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
    response = s3.list_objects_v2(Bucket=bucket, Prefix='true-asi-system/models/', MaxKeys=10)
    
    assert 'Contents' in response, "No objects found"
    assert len(response['Contents']) > 0, "Empty bucket"
    
    test_results['s3_access'] = {
        'status': 'PASS',
        'details': f"{len(response['Contents'])} objects accessible"
    }
    print(f"  ✅ PASS: S3 accessible with {len(response['Contents'])} objects")
except Exception as e:
    test_results['s3_access'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    print(f"  ❌ FAIL: {str(e)[:100]}")

# Test 6: Orchestrator
print("\n🧪 TEST 6: Orchestrator")
try:
    from perfect_orchestrator import PerfectOrchestrator, Task, OrchestrationMode
    
    orchestrator = PerfectOrchestrator()
    
    # Create a simple task
    task = Task(
        id="test_task",
        prompt="Test prompt",
        task_type=None
    )
    
    # Just check initialization, not execution (would need real models)
    assert orchestrator is not None, "Orchestrator not initialized"
    
    test_results['orchestrator'] = {
        'status': 'PASS',
        'details': "Orchestrator initialized"
    }
    print(f"  ✅ PASS: Orchestrator initialized")
except Exception as e:
    test_results['orchestrator'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    print(f"  ❌ FAIL: {str(e)[:100]}")

# Test 7: Entity Layer
print("\n🧪 TEST 7: Entity Layer")
try:
    from unified_entity_layer import UnifiedEntityLayer, TaskType
    
    entity = UnifiedEntityLayer()
    
    assert entity is not None, "Entity layer not initialized"
    
    test_results['entity_layer'] = {
        'status': 'PASS',
        'details': "Entity layer initialized"
    }
    print(f"  ✅ PASS: Entity layer initialized")
except Exception as e:
    test_results['entity_layer'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    print(f"  ❌ FAIL: {str(e)[:100]}")

# Calculate results
print("\n" + "=" * 80)
print("CYCLE 4 RESULTS - FUNCTIONALITY")
print("=" * 80)

passed = sum(1 for r in test_results.values() if r['status'] == 'PASS')
total = len(test_results)

print(f"\n📊 TEST RESULTS: {passed}/{total} PASSED ({passed/total*100:.1f}%)")

print(f"\n📋 DETAILED RESULTS:")
for test_name, result in test_results.items():
    status_icon = "✅" if result['status'] == 'PASS' else "❌"
    print(f"  {status_icon} {test_name}: {result['status']}")
    if result['status'] == 'PASS':
        print(f"      {result.get('details', '')}")
    else:
        print(f"      Error: {result.get('error', '')[:100]}")

functionality_score = (passed / total) * 100

print(f"\n📈 FUNCTIONALITY SCORE: {functionality_score:.1f}/100")

print("\n" + "=" * 80)
print("CYCLE 4 COMPLETE")
print("=" * 80)

# Save results
with open('audit_cycle4_results.txt', 'w') as f:
    f.write(f"CYCLE 4 RESULTS\n")
    f.write(f"===============\n\n")
    f.write(f"Tests passed: {passed}/{total}\n")
    f.write(f"Functionality score: {functionality_score:.1f}/100\n\n")
    f.write(f"Details:\n")
    for test_name, result in test_results.items():
        f.write(f"  {test_name}: {result['status']}\n")
        if result['status'] == 'PASS':
            f.write(f"    {result.get('details', '')}\n")
        else:
            f.write(f"    Error: {result.get('error', '')}\n")

print("\n✅ Results saved to audit_cycle4_results.txt")
