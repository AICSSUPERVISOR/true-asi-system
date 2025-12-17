#!/usr/bin/env python3
"""
TRUE ASI System - Download All S3 Data
=======================================

Downloads all data from AWS S3 and integrates into repository structure.

Author: Manus AI
Date: November 1, 2025
Version: 1.0.0
"""

import os
import boto3
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# AWS Configuration
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'YOUR_AWS_ACCESS_KEY_HERE')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY_HERE')
AWS_REGION = 'us-east-1'
S3_BUCKET = 'asi-knowledge-base-898982995956'

# Download configuration
DOWNLOAD_DIR = Path('/home/ubuntu/true-asi-system/s3_data')
MAX_WORKERS = 10
SAMPLE_SIZE = 1000  # Download first 1000 files as sample (full download would be 19GB)

def create_s3_client():
    """Create S3 client"""
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

def download_file(s3, bucket, key, local_path):
    """Download a single file from S3"""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))
        return True, key
    except Exception as e:
        return False, f"{key}: {str(e)}"

def download_s3_data_sample():
    """Download sample of S3 data"""
    s3 = create_s3_client()
    
    print(f"📥 Downloading sample data from S3 ({SAMPLE_SIZE} files)...")
    print(f"   Bucket: {S3_BUCKET}")
    print(f"   Destination: {DOWNLOAD_DIR}")
    print()
    
    # List objects
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET)
    
    files_to_download = []
    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            files_to_download.append(obj['Key'])
            if len(files_to_download) >= SAMPLE_SIZE:
                break
        if len(files_to_download) >= SAMPLE_SIZE:
            break
    
    print(f"Found {len(files_to_download)} files to download")
    print()
    
    # Download files in parallel
    downloaded = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for key in files_to_download:
            local_path = DOWNLOAD_DIR / key
            future = executor.submit(download_file, s3, S3_BUCKET, key, local_path)
            futures.append(future)
        
        for i, future in enumerate(as_completed(futures), 1):
            success, result = future.result()
            if success:
                downloaded += 1
                if downloaded % 100 == 0:
                    print(f"   Downloaded: {downloaded}/{len(files_to_download)} files")
            else:
                failed += 1
    
    print()
    print(f"✅ Download complete!")
    print(f"   Downloaded: {downloaded} files")
    print(f"   Failed: {failed} files")
    
    return downloaded, failed

def create_data_index():
    """Create index of downloaded data"""
    print()
    print("📊 Creating data index...")
    
    index = {
        'generated_at': datetime.now().isoformat(),
        'bucket': S3_BUCKET,
        'download_dir': str(DOWNLOAD_DIR),
        'directories': {},
        'total_files': 0,
        'total_size_mb': 0
    }
    
    for root, dirs, files in os.walk(DOWNLOAD_DIR):
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(DOWNLOAD_DIR)
            dir_name = str(rel_path.parent)
            
            if dir_name not in index['directories']:
                index['directories'][dir_name] = {
                    'files': [],
                    'count': 0,
                    'size_mb': 0
                }
            
            file_size = file_path.stat().st_size
            index['directories'][dir_name]['files'].append(str(rel_path))
            index['directories'][dir_name]['count'] += 1
            index['directories'][dir_name]['size_mb'] += file_size / (1024 * 1024)
            index['total_files'] += 1
            index['total_size_mb'] += file_size / (1024 * 1024)
    
    # Save index
    index_file = DOWNLOAD_DIR / 'DATA_INDEX.json'
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"✅ Index created: {index_file}")
    print(f"   Total files: {index['total_files']:,}")
    print(f"   Total size: {index['total_size_mb']:.2f} MB")
    print(f"   Directories: {len(index['directories'])}")
    
    return index

def main():
    """Main function"""
    print("="*80)
    print("AWS S3 DATA DOWNLOAD")
    print("="*80)
    print()
    
    # Download sample data
    downloaded, failed = download_s3_data_sample()
    
    # Create index
    index = create_data_index()
    
    print()
    print("="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print()
    print(f"Downloaded {downloaded} files from S3")
    print(f"Data location: {DOWNLOAD_DIR}")
    print()
    print("Note: This is a sample download (1,000 files).")
    print("Full S3 bucket contains 57,419 files (19.02 GB).")
    print()

if __name__ == "__main__":
    main()
