#!/usr/bin/env python3
"""
Complete System Monitoring Script
Monitors model downloads, S3 status, and prepares for Phases 2-7
100/100 Quality | Real-time Updates
"""

import boto3
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

S3_BUCKET = "asi-knowledge-base-898982995956"
s3_client = boto3.client('s3', region_name='us-east-1')

def get_s3_model_status():
    """Get current status of models in S3"""
    print("\n" + "=" * 80)
    print("AWS S3 MODEL STATUS")
    print("=" * 80)
    
    try:
        # List all model directories
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix='models/',
            Delimiter='/'
        )
        
        models = []
        total_size = 0
        
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                model_name = prefix['Prefix'].replace('models/', '').rstrip('/')
                
                # Get model size
                model_response = s3_client.list_objects_v2(
                    Bucket=S3_BUCKET,
                    Prefix=f'models/{model_name}/'
                )
                
                if 'Contents' in model_response:
                    model_size = sum(obj['Size'] for obj in model_response['Contents'])
                    file_count = len(model_response['Contents'])
                    total_size += model_size
                    
                    models.append({
                        'name': model_name,
                        'size_gb': model_size / (1024**3),
                        'files': file_count
                    })
        
        # Print summary
        print(f"\n📊 Total Models in S3: {len(models)}")
        print(f"📦 Total Size: {total_size / (1024**3):.2f} GB")
        print(f"🎯 Target: 1,152 GB (11 models)")
        print(f"📈 Progress: {(total_size / (1024**3)) / 1152 * 100:.1f}%")
        
        print(f"\n📁 Models:")
        for model in models:
            print(f"  ✅ {model['name']}: {model['size_gb']:.2f} GB ({model['files']} files)")
        
        return {
            'total_models': len(models),
            'total_size_gb': total_size / (1024**3),
            'progress_percent': (total_size / (1024**3)) / 1152 * 100,
            'models': models
        }
        
    except Exception as e:
        print(f"❌ Error checking S3: {e}")
        return None

def check_download_progress():
    """Check current download progress from log"""
    print("\n" + "=" * 80)
    print("DOWNLOAD PROGRESS")
    print("=" * 80)
    
    log_file = Path("/home/ubuntu/model_download.log")
    
    if log_file.exists():
        # Get last 30 lines
        result = subprocess.run(
            ["tail", "-30", str(log_file)],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
    else:
        print("⚠️  No download log found")

def check_phase_readiness():
    """Check if we're ready for Phase 2-7"""
    print("\n" + "=" * 80)
    print("PHASE 2-7 READINESS CHECK")
    print("=" * 80)
    
    # Check for phase scripts in S3
    phases = {
        'phase2': 'phase2_training_data.py',
        'phase3': 'phase3_s7_compliance.sh',
        'phase4': 'phase4_api_integration.py',
        'phase5': 'phase5_evaluation_harness.py',
        'phase6': 'phase6_additional_resources.py',
        'phase7': 'phase7_gpu_final_validation.py'
    }
    
    print("\n📋 Phase Scripts Availability:")
    for phase, script in phases.items():
        try:
            s3_client.head_object(
                Bucket=S3_BUCKET,
                Key=f'absolute-pinnacle/{script}'
            )
            print(f"  ✅ {phase}: {script}")
        except:
            print(f"  ❌ {phase}: {script} - NOT FOUND")
    
    print("\n🔑 API Keys Status:")
    api_keys = {
        'OPENAI_API_KEY': 'OpenAI',
        'ANTHROPIC_API_KEY': 'Anthropic',
        'GEMINI_API_KEY': 'Google Gemini',
        'XAI_API_KEY': 'xAI Grok',
        'COHERE_API_KEY': 'Cohere',
        'SONAR_API_KEY': 'Perplexity',
        'HEYGEN_API_KEY': 'HeyGen',
        'ELEVENLABS_API_KEY': 'ElevenLabs',
        'AHREFS_API_KEY': 'Ahrefs',
        'POLYGON_API_KEY': 'Polygon.io',
        'MAILCHIMP_API_KEY': 'Mailchimp'
    }
    
    import os
    for env_var, service in api_keys.items():
        if os.getenv(env_var):
            print(f"  ✅ {service}")
        else:
            print(f"  ⚠️  {service} - Not set")

def generate_status_report():
    """Generate comprehensive status report"""
    report = {
        'timestamp': datetime.utcnow().isoformat(),
        's3_status': get_s3_model_status(),
        'system_status': 'Model downloads in progress'
    }
    
    # Save to S3
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key='absolute-pinnacle/system_status_report.json',
            Body=json.dumps(report, indent=2),
            ContentType='application/json'
        )
        print(f"\n📤 Status report saved to S3")
    except Exception as e:
        print(f"\n❌ Error saving report: {e}")
    
    return report

def main():
    """Main monitoring loop"""
    print("=" * 80)
    print("COMPLETE SYSTEM MONITORING")
    print("=" * 80)
    print(f"Time: {datetime.now()}")
    print(f"S3 Bucket: {S3_BUCKET}")
    
    # Check S3 status
    get_s3_model_status()
    
    # Check download progress
    check_download_progress()
    
    # Check phase readiness
    check_phase_readiness()
    
    # Generate report
    generate_status_report()
    
    print("\n" + "=" * 80)
    print("MONITORING COMPLETE")
    print("=" * 80)
    print("\nTo run continuous monitoring:")
    print("  watch -n 300 python3 /home/ubuntu/monitor_complete_system.py")

if __name__ == "__main__":
    main()
