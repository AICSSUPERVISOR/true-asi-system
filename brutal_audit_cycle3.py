"""
BRUTAL AUDIT - CYCLE 3: S3 & Data
==================================

Verify actual files and models in AWS S3.
100% factual results only.
"""

import boto3
import os
from collections import defaultdict

print("=" * 80)
print("BRUTAL AUDIT - CYCLE 3: S3 & DATA")
print("=" * 80)

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='us-east-1'
)

bucket = 'asi-knowledge-base-898982995956'

print(f"\n🔍 SCANNING S3 BUCKET: {bucket}")
print("This may take a while...")

# Scan models directory
print("\n1. MODELS DIRECTORY (true-asi-system/models/):")
try:
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix='true-asi-system/models/')
    
    model_files = {}
    total_size = 0
    file_count = 0
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                size = obj['Size']
                
                # Extract model name
                parts = key.split('/')
                if len(parts) >= 4:
                    model_name = parts[3]
                    if model_name not in model_files:
                        model_files[model_name] = {'files': 0, 'size': 0}
                    model_files[model_name]['files'] += 1
                    model_files[model_name]['size'] += size
                
                total_size += size
                file_count += 1
    
    print(f"  ✅ Total models found: {len(model_files)}")
    print(f"  ✅ Total files: {file_count}")
    print(f"  ✅ Total size: {total_size / (1024**3):.2f} GB")
    
    # Show top 10 models by size
    sorted_models = sorted(model_files.items(), key=lambda x: x[1]['size'], reverse=True)
    print(f"\n  Top 10 models by size:")
    for i, (name, info) in enumerate(sorted_models[:10], 1):
        size_gb = info['size'] / (1024**3)
        print(f"    {i}. {name}: {info['files']} files, {size_gb:.2f} GB")
    
except Exception as e:
    print(f"  ❌ Error scanning models: {str(e)[:200]}")
    model_files = {}
    total_size = 0
    file_count = 0

# Scan code directory
print("\n2. CODE DIRECTORY (true-asi-system/code/):")
try:
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix='true-asi-system/code/')
    
    code_files = 0
    code_size = 0
    file_types = defaultdict(int)
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                size = obj['Size']
                
                # Get file extension
                if '.' in key:
                    ext = key.split('.')[-1]
                    file_types[ext] += 1
                
                code_files += 1
                code_size += size
    
    print(f"  ✅ Total code files: {code_files}")
    print(f"  ✅ Total size: {code_size / (1024**2):.2f} MB")
    print(f"\n  Files by type:")
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    .{ext}: {count} files")
    
except Exception as e:
    print(f"  ❌ Error scanning code: {str(e)[:200]}")
    code_files = 0
    code_size = 0

# Check download progress
print("\n3. DOWNLOAD PROGRESS:")
try:
    # Check if downloader is still running
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    downloader_running = 'direct_to_s3_downloader' in result.stdout
    
    print(f"  {'✅' if downloader_running else '❌'} Downloader running: {downloader_running}")
    
    # Check progress file
    try:
        import json
        from pathlib import Path
        progress_file = Path('direct_s3_progress.json')
        if progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)
            downloaded = progress.get('downloaded', [])
            print(f"  ✅ Models downloaded: {len(downloaded)}")
        else:
            print(f"  ⚠️  Progress file not found")
    except Exception as e:
        print(f"  ⚠️  Could not read progress: {str(e)[:100]}")
    
except Exception as e:
    print(f"  ❌ Error checking progress: {str(e)[:200]}")

# Verify claimed numbers
print("\n" + "=" * 80)
print("CYCLE 3 RESULTS - S3 & DATA")
print("=" * 80)

print(f"\n📊 ACTUAL S3 CONTENTS:")
print(f"  Models in S3: {len(model_files)}")
print(f"  Model files: {file_count}")
print(f"  Model storage: {total_size / (1024**3):.2f} GB")
print(f"  Code files in S3: {code_files}")
print(f"  Code storage: {code_size / (1024**2):.2f} MB")

print(f"\n📋 CLAIMED VS ACTUAL:")
print(f"  Claimed: 296 models total")
print(f"  Actual in S3: {len(model_files)} models")
print(f"  Difference: {296 - len(model_files)} models")

print(f"\n  Claimed: 34+ models in S3")
print(f"  Actual: {len(model_files)} models")
print(f"  {'✅ MATCH' if len(model_files) >= 34 else '❌ MISMATCH'}")

# Calculate S3 score
s3_score = 100
if len(model_files) < 34:
    s3_score -= 30
if code_files < 300:
    s3_score -= 20
if total_size < 100 * (1024**3):  # Less than 100 GB
    s3_score -= 10

print(f"\n📈 S3 DATA SCORE: {s3_score:.1f}/100")

print("\n" + "=" * 80)
print("CYCLE 3 COMPLETE")
print("=" * 80)

# Save results
with open('audit_cycle3_results.txt', 'w') as f:
    f.write(f"CYCLE 3 RESULTS\n")
    f.write(f"===============\n\n")
    f.write(f"Models in S3: {len(model_files)}\n")
    f.write(f"Model files: {file_count}\n")
    f.write(f"Model storage: {total_size / (1024**3):.2f} GB\n")
    f.write(f"Code files: {code_files}\n")
    f.write(f"Code storage: {code_size / (1024**2):.2f} MB\n")
    f.write(f"S3 score: {s3_score:.1f}/100\n")

print("\n✅ Results saved to audit_cycle3_results.txt")
