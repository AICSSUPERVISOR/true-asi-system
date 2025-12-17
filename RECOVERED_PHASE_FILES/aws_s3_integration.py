#!/usr/bin/env python3.11
"""
TRUE ASI SYSTEM - AWS S3 Integration & Data Organization
Comprehensive AWS S3 backend integration with organized structure
"""

import os
import boto3
import json
import hashlib
from datetime import datetime
from pathlib import Path
import zipfile
import tarfile
import gzip
from typing import Dict, List, Any
import subprocess

class AWSS3Integration:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'asi-knowledge-base-898982995956'
        self.region = 'us-east-1'
        
        self.upload_stats = {
            "total_files": 0,
            "total_bytes": 0,
            "successful_uploads": 0,
            "failed_uploads": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Define organized S3 structure
        self.s3_structure = {
            "core-system/": {
                "s7-architecture/": "S-7 layer files and architecture",
                "agents/": "Agent system files",
                "models/": "Model files and configurations",
                "infrastructure/": "Infrastructure code",
                "training/": "Training pipelines and scripts",
                "memory/": "Memory system components",
                "tools/": "Tool execution systems",
                "alignment/": "Alignment and safety systems"
            },
            "knowledge-base/": {
                "llm-models/": "LLM model catalog and metadata",
                "repositories/": "Integrated repositories",
                "documentation/": "System documentation",
                "research/": "Research papers and references"
            },
            "industry-modules/": {
                "medical/": "Medical AI system",
                "finance/": "Finance AI system",
                "legal/": "Legal AI system",
                "education/": "Education AI system",
                "manufacturing/": "Manufacturing AI system",
                # ... 45 more industries
            },
            "deployments/": {
                "production/": "Production deployments",
                "staging/": "Staging deployments",
                "testing/": "Test deployments"
            },
            "training-data/": {
                "raw/": "Raw training data",
                "processed/": "Processed training data",
                "embeddings/": "Embedding vectors"
            },
            "backups/": {
                "daily/": "Daily backups",
                "weekly/": "Weekly backups",
                "critical/": "Critical system backups"
            },
            "logs/": {
                "system/": "System logs",
                "agent/": "Agent logs",
                "api/": "API logs"
            }
        }
    
    def create_s3_structure(self):
        """Create organized folder structure in S3"""
        print("Creating organized S3 bucket structure...")
        
        folders_created = 0
        
        for parent_folder, subfolders in self.s3_structure.items():
            # Create parent folder marker
            self._create_folder_marker(parent_folder)
            folders_created += 1
            
            if isinstance(subfolders, dict):
                for subfolder, description in subfolders.items():
                    full_path = f"{parent_folder}{subfolder}"
                    self._create_folder_marker(full_path)
                    
                    # Create README for each folder
                    readme_content = f"# {subfolder}\n\n{description}\n\nCreated: {datetime.now().isoformat()}\n"
                    self._upload_content(
                        content=readme_content.encode('utf-8'),
                        s3_key=f"{full_path}README.md"
                    )
                    folders_created += 1
        
        print(f"✅ Created {folders_created} folders in S3")
        return folders_created
    
    def _create_folder_marker(self, folder_path: str):
        """Create a folder marker in S3 (zero-byte object ending with /)"""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=folder_path if folder_path.endswith('/') else f"{folder_path}/",
                Body=b''
            )
        except Exception as e:
            print(f"Warning: Could not create folder marker {folder_path}: {e}")
    
    def _upload_content(self, content: bytes, s3_key: str):
        """Upload content to S3"""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content
            )
            return True
        except Exception as e:
            print(f"Error uploading {s3_key}: {e}")
            return False
    
    def upload_file(self, local_path: str, s3_key: str, compress: bool = False) -> bool:
        """Upload a file to S3 with optional compression"""
        try:
            if not os.path.exists(local_path):
                print(f"File not found: {local_path}")
                return False
            
            file_size = os.path.getsize(local_path)
            
            if compress and local_path.endswith('.py'):
                # Compress Python files
                with open(local_path, 'rb') as f_in:
                    compressed_data = gzip.compress(f_in.read())
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=f"{s3_key}.gz",
                    Body=compressed_data,
                    Metadata={
                        'original_size': str(file_size),
                        'compressed': 'true',
                        'upload_time': datetime.now().isoformat()
                    }
                )
                s3_key = f"{s3_key}.gz"
            else:
                # Upload without compression
                self.s3_client.upload_file(
                    local_path,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={
                        'Metadata': {
                            'upload_time': datetime.now().isoformat(),
                            'file_size': str(file_size)
                        }
                    }
                )
            
            self.upload_stats["total_files"] += 1
            self.upload_stats["total_bytes"] += file_size
            self.upload_stats["successful_uploads"] += 1
            
            return True
        
        except Exception as e:
            print(f"Error uploading {local_path} to {s3_key}: {e}")
            self.upload_stats["failed_uploads"] += 1
            return False
    
    def upload_directory(self, local_dir: str, s3_prefix: str, compress_py: bool = True):
        """Upload entire directory to S3"""
        print(f"Uploading directory: {local_dir} -> s3://{self.bucket_name}/{s3_prefix}")
        
        if not os.path.exists(local_dir):
            print(f"Directory not found: {local_dir}")
            return
        
        uploaded = 0
        skipped = 0
        
        for root, dirs, files in os.walk(local_dir):
            # Skip .git directories
            if '.git' in root:
                continue
            
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}{relative_path}".replace('\\', '/')
                
                # Determine if we should compress
                should_compress = compress_py and file.endswith('.py')
                
                if self.upload_file(local_path, s3_key, compress=should_compress):
                    uploaded += 1
                    if uploaded % 50 == 0:
                        print(f"  Uploaded {uploaded} files...")
                else:
                    skipped += 1
        
        print(f"✅ Uploaded {uploaded} files, skipped {skipped}")
    
    def upload_github_repository(self):
        """Upload complete GitHub repository to S3"""
        print("\n" + "="*60)
        print("UPLOADING GITHUB REPOSITORY TO S3")
        print("="*60)
        
        repo_path = "/home/ubuntu/true-asi-system"
        s3_prefix = "core-system/"
        
        if not os.path.exists(repo_path):
            print(f"Repository not found: {repo_path}")
            return
        
        # Upload different components to organized locations
        components = {
            "models/s7_layers/": "core-system/s7-architecture/",
            "agents/": "core-system/agents/",
            "models/": "core-system/models/",
            "infrastructure/": "core-system/infrastructure/",
            "training/": "core-system/training/",
            "monitoring/": "core-system/monitoring/",
            "deployment/": "deployments/production/",
            "docs/": "knowledge-base/documentation/"
        }
        
        # Upload root level files
        for file in os.listdir(repo_path):
            file_path = os.path.join(repo_path, file)
            if os.path.isfile(file_path) and not file.startswith('.'):
                s3_key = f"core-system/{file}"
                self.upload_file(file_path, s3_key)
        
        # Upload organized components
        for local_subdir, s3_dest in components.items():
            local_path = os.path.join(repo_path, local_subdir)
            if os.path.exists(local_path):
                self.upload_directory(local_path, s3_dest)
    
    def upload_zip_archives(self):
        """Upload ZIP archives to S3"""
        print("\n" + "="*60)
        print("UPLOADING ZIP ARCHIVES TO S3")
        print("="*60)
        
        zip_files = [
            "/home/ubuntu/ASI-Production-Grade-System-112.zip",
            "/home/ubuntu/ASI-Production-Grade-System-113.zip",
            "/home/ubuntu/ASI-Production-Grade-System-115.zip"
        ]
        
        for zip_file in zip_files:
            if os.path.exists(zip_file):
                filename = os.path.basename(zip_file)
                s3_key = f"backups/critical/{filename}"
                
                print(f"Uploading {filename}...")
                if self.upload_file(zip_file, s3_key, compress=False):
                    print(f"✅ Uploaded {filename}")
    
    def upload_build_artifacts(self):
        """Upload build artifacts and generated files"""
        print("\n" + "="*60)
        print("UPLOADING BUILD ARTIFACTS TO S3")
        print("="*60)
        
        build_dir = "/home/ubuntu/true-asi-build"
        
        if os.path.exists(build_dir):
            self.upload_directory(build_dir, "deployments/production/build/")
    
    def create_manifest(self) -> Dict[str, Any]:
        """Create a manifest of all uploaded files"""
        print("\nCreating upload manifest...")
        
        manifest = {
            "upload_date": datetime.now().isoformat(),
            "bucket": self.bucket_name,
            "region": self.region,
            "statistics": self.upload_stats,
            "structure": self.s3_structure
        }
        
        # Save manifest locally
        manifest_path = "/home/ubuntu/true-asi-build/s3_upload_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Upload manifest to S3
        self.upload_file(manifest_path, "deployments/production/s3_upload_manifest.json")
        
        print(f"✅ Manifest created and uploaded")
        return manifest
    
    def verify_uploads(self) -> Dict[str, Any]:
        """Verify all uploads were successful"""
        print("\n" + "="*60)
        print("VERIFYING S3 UPLOADS")
        print("="*60)
        
        try:
            # List all objects in bucket
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=1000
            )
            
            total_objects = response.get('KeyCount', 0)
            total_size = sum(obj.get('Size', 0) for obj in response.get('Contents', []))
            
            verification = {
                "status": "success",
                "total_objects": total_objects,
                "total_size_mb": round(total_size / (1024*1024), 2),
                "verification_time": datetime.now().isoformat()
            }
            
            print(f"✅ Verification complete:")
            print(f"   Total objects: {total_objects}")
            print(f"   Total size: {verification['total_size_mb']} MB")
            
            return verification
        
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def setup_continuous_autosave(self):
        """Set up continuous auto-save system"""
        print("\n" + "="*60)
        print("SETTING UP CONTINUOUS AUTO-SAVE")
        print("="*60)
        
        autosave_script = """#!/usr/bin/env python3.11
import time
import os
import boto3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class S3AutoSaveHandler(FileSystemEventHandler):
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket = 'asi-knowledge-base-898982995956'
        self.last_upload = {}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Only auto-save Python files
        if not event.src_path.endswith('.py'):
            return
        
        # Debounce: Don't upload same file within 10 seconds
        current_time = time.time()
        if event.src_path in self.last_upload:
            if current_time - self.last_upload[event.src_path] < 10:
                return
        
        try:
            # Determine S3 key based on file location
            if 'true-asi-system' in event.src_path:
                relative_path = event.src_path.split('true-asi-system/')[1]
                s3_key = f"core-system/{relative_path}"
            else:
                s3_key = f"backups/autosave/{os.path.basename(event.src_path)}"
            
            # Upload to S3
            self.s3_client.upload_file(event.src_path, self.bucket, s3_key)
            self.last_upload[event.src_path] = current_time
            print(f"Auto-saved: {s3_key}")
        
        except Exception as e:
            print(f"Auto-save error: {e}")

if __name__ == "__main__":
    print("Starting continuous auto-save system...")
    
    event_handler = S3AutoSaveHandler()
    observer = Observer()
    
    # Watch directories
    watch_dirs = [
        "/home/ubuntu/true-asi-system",
        "/home/ubuntu/true-asi-build"
    ]
    
    for watch_dir in watch_dirs:
        if os.path.exists(watch_dir):
            observer.schedule(event_handler, watch_dir, recursive=True)
            print(f"Watching: {watch_dir}")
    
    observer.start()
    
    try:
        while True:
            time.sleep(60)  # Keep running
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()
"""
        
        autosave_path = "/home/ubuntu/true-asi-build/continuous_autosave.py"
        with open(autosave_path, 'w') as f:
            f.write(autosave_script)
        
        os.chmod(autosave_path, 0o755)
        
        # Upload autosave script to S3
        self.upload_file(autosave_path, "core-system/infrastructure/continuous_autosave.py")
        
        print(f"✅ Auto-save script created: {autosave_path}")
        print("   To start: python3.11 /home/ubuntu/true-asi-build/continuous_autosave.py &")
    
    def run_full_integration(self):
        """Run complete AWS S3 integration"""
        print("\n" + "="*80)
        print("TRUE ASI SYSTEM - AWS S3 INTEGRATION")
        print("="*80)
        print(f"Bucket: {self.bucket_name}")
        print(f"Region: {self.region}")
        print(f"Start Time: {self.upload_stats['start_time']}")
        print("="*80)
        
        # Step 1: Create organized structure
        self.create_s3_structure()
        
        # Step 2: Upload GitHub repository
        self.upload_github_repository()
        
        # Step 3: Upload ZIP archives
        self.upload_zip_archives()
        
        # Step 4: Upload build artifacts
        self.upload_build_artifacts()
        
        # Step 5: Create manifest
        manifest = self.create_manifest()
        
        # Step 6: Verify uploads
        verification = self.verify_uploads()
        
        # Step 7: Set up continuous auto-save
        self.setup_continuous_autosave()
        
        # Final summary
        print("\n" + "="*80)
        print("AWS S3 INTEGRATION COMPLETE")
        print("="*80)
        print(f"Total files uploaded: {self.upload_stats['successful_uploads']}")
        print(f"Total bytes uploaded: {self.upload_stats['total_bytes']:,}")
        print(f"Total size: {round(self.upload_stats['total_bytes'] / (1024*1024*1024), 2)} GB")
        print(f"Failed uploads: {self.upload_stats['failed_uploads']}")
        print(f"End Time: {datetime.now().isoformat()}")
        print("="*80)
        
        return {
            "upload_stats": self.upload_stats,
            "manifest": manifest,
            "verification": verification
        }

if __name__ == "__main__":
    integrator = AWSS3Integration()
    result = integrator.run_full_integration()
    
    # Save result
    result_path = "/home/ubuntu/true-asi-build/aws_integration_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Integration result saved to: {result_path}")
