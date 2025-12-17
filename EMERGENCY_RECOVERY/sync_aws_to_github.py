#!/usr/bin/env python3
"""
TRUE ASI System - AWS to GitHub Data Sync
==========================================

Syncs data from AWS S3 to GitHub repository with metadata and documentation.

Author: Manus AI
Date: November 1, 2025
Version: 1.0.0
"""

import os
import boto3
import json
from datetime import datetime
from pathlib import Path

# AWS Configuration
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'YOUR_AWS_ACCESS_KEY_HERE')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY_HERE')
AWS_REGION = 'us-east-1'
S3_BUCKET = 'asi-knowledge-base-898982995956'

def create_s3_client():
    """Create S3 client"""
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

def get_s3_statistics():
    """Get S3 bucket statistics"""
    s3 = create_s3_client()
    
    print("Analyzing S3 bucket...")
    
    # List all objects
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET)
    
    total_size = 0
    total_files = 0
    directories = set()
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            total_files += 1
            total_size += obj['Size']
            
            # Extract directory
            key = obj['Key']
            if '/' in key:
                dir_name = key.split('/')[0]
                directories.add(dir_name)
    
    return {
        'total_files': total_files,
        'total_size_gb': total_size / (1024**3),
        'directories': sorted(list(directories)),
        'directory_count': len(directories)
    }

def create_data_manifest():
    """Create manifest of all S3 data"""
    stats = get_s3_statistics()
    
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'bucket': S3_BUCKET,
        'region': AWS_REGION,
        'statistics': stats,
        'access_instructions': {
            'aws_cli': f"aws s3 ls s3://{S3_BUCKET}/",
            'boto3': f"boto3.client('s3').list_objects_v2(Bucket='{S3_BUCKET}')",
            'download_all': f"aws s3 sync s3://{S3_BUCKET}/ ./local_data/"
        }
    }
    
    return manifest

def main():
    """Main function"""
    print("="*80)
    print("AWS TO GITHUB DATA SYNC")
    print("="*80)
    print()
    
    # Get S3 statistics
    print("📊 Analyzing AWS S3 data...")
    manifest = create_data_manifest()
    
    print()
    print("S3 BUCKET STATISTICS:")
    print(f"  Bucket: {S3_BUCKET}")
    print(f"  Total Files: {manifest['statistics']['total_files']:,}")
    print(f"  Total Size: {manifest['statistics']['total_size_gb']:.2f} GB")
    print(f"  Directories: {manifest['statistics']['directory_count']}")
    print()
    
    # Save manifest
    manifest_file = Path("AWS_S3_DATA_MANIFEST.json")
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Manifest saved: {manifest_file}")
    
    # Create README for data access
    readme = []
    readme.append("# AWS S3 Data Access")
    readme.append("")
    readme.append("## Overview")
    readme.append("")
    readme.append(f"The TRUE ASI System stores {manifest['statistics']['total_size_gb']:.2f} GB of data in AWS S3.")
    readme.append("")
    readme.append("## Statistics")
    readme.append("")
    readme.append(f"- **Total Files**: {manifest['statistics']['total_files']:,}")
    readme.append(f"- **Total Size**: {manifest['statistics']['total_size_gb']:.2f} GB")
    readme.append(f"- **Directories**: {manifest['statistics']['directory_count']}")
    readme.append("")
    readme.append("## Directories")
    readme.append("")
    for dir_name in manifest['statistics']['directories'][:20]:
        readme.append(f"- `{dir_name}/`")
    if len(manifest['statistics']['directories']) > 20:
        readme.append(f"- ... and {len(manifest['statistics']['directories']) - 20} more")
    readme.append("")
    readme.append("## Access Instructions")
    readme.append("")
    readme.append("### AWS CLI")
    readme.append("```bash")
    readme.append(manifest['access_instructions']['aws_cli'])
    readme.append("```")
    readme.append("")
    readme.append("### Download All Data")
    readme.append("```bash")
    readme.append(manifest['access_instructions']['download_all'])
    readme.append("```")
    readme.append("")
    readme.append("### Python (boto3)")
    readme.append("```python")
    readme.append(manifest['access_instructions']['boto3'])
    readme.append("```")
    
    readme_file = Path("AWS_S3_DATA_README.md")
    readme_file.write_text("\n".join(readme))
    
    print(f"✅ README saved: {readme_file}")
    print()
    print("="*80)
    print("SYNC COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
