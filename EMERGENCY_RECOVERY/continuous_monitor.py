"""
Continuous Download & S3 Monitor
Tracks progress and saves status to S3 continuously
"""

import boto3
import os
import json
import time
from datetime import datetime
from collections import defaultdict

# AWS Configuration
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='us-east-1'
)

bucket = 'asi-knowledge-base-898982995956'

def get_s3_status():
    """Get current S3 model status."""
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix='true-asi-system/models/')
    
    models = set()
    total_files = 0
    total_size = 0
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                size = obj['Size']
                parts = key.split('/')
                if len(parts) >= 4:
                    models.add(parts[3])
                total_files += 1
                total_size += size
    
    return {
        'models': len(models),
        'files': total_files,
        'size_gb': total_size / (1024**3),
        'timestamp': datetime.now().isoformat()
    }

def get_download_progress():
    """Get downloader progress."""
    try:
        with open('direct_s3_progress.json', 'r') as f:
            return json.load(f)
    except:
        return {'downloaded': [], 'failed': [], 'in_progress': []}

def save_status_to_s3(status):
    """Save status report to S3."""
    report = f"""# TRUE ASI SYSTEM - CONTINUOUS MONITORING
**Timestamp:** {status['timestamp']}

## S3 Status
- Models: {status['s3']['models']}
- Files: {status['s3']['files']}
- Size: {status['s3']['size_gb']:.2f} GB

## Download Progress
- Downloaded: {len(status['download']['downloaded'])}
- Failed: {len(status['download']['failed'])}
- In Progress: {len(status['download']['in_progress'])}

## Overall Progress
- Total Target: 296 models
- Completion: {status['s3']['models']/296*100:.1f}%
- Remaining: {296 - status['s3']['models']} models
"""
    
    # Save to S3
    s3.put_object(
        Bucket=bucket,
        Key='true-asi-system/status/latest_status.md',
        Body=report.encode('utf-8')
    )
    
    # Save JSON version
    s3.put_object(
        Bucket=bucket,
        Key='true-asi-system/status/latest_status.json',
        Body=json.dumps(status, indent=2).encode('utf-8')
    )

def monitor():
    """Run continuous monitoring."""
    print("=" * 80)
    print("CONTINUOUS MONITORING STARTED")
    print("=" * 80)
    
    baseline = get_s3_status()
    print(f"\n📊 Baseline: {baseline['models']} models, {baseline['size_gb']:.2f} GB")
    
    iteration = 0
    
    while True:
        iteration += 1
        time.sleep(300)  # Check every 5 minutes
        
        # Get current status
        s3_status = get_s3_status()
        download_status = get_download_progress()
        
        # Calculate progress
        models_added = s3_status['models'] - baseline['models']
        gb_added = s3_status['size_gb'] - baseline['size_gb']
        
        status = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            's3': s3_status,
            'download': download_status,
            'progress': {
                'models_added': models_added,
                'gb_added': gb_added,
                'completion_pct': s3_status['models'] / 296 * 100
            }
        }
        
        # Print status
        print(f"\n{'='*80}")
        print(f"Iteration {iteration} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        print(f"📊 S3: {s3_status['models']} models, {s3_status['size_gb']:.2f} GB")
        print(f"📈 Progress: +{models_added} models, +{gb_added:.2f} GB since start")
        print(f"✅ Downloaded: {len(download_status['downloaded'])}")
        print(f"⏳ Completion: {status['progress']['completion_pct']:.1f}%")
        print(f"🎯 Remaining: {296 - s3_status['models']} models")
        
        # Save to S3
        try:
            save_status_to_s3(status)
            print(f"💾 Status saved to S3")
        except Exception as e:
            print(f"⚠️ Failed to save to S3: {e}")
        
        # Check if complete
        if s3_status['models'] >= 296:
            print(f"\n🎉 COMPLETE! All 296 models downloaded!")
            break

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitoring stopped by user")
    except Exception as e:
        print(f"\n\n❌ Monitoring error: {e}")
