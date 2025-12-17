#!/usr/bin/env python3
"""
Continuous S3 Auto-Save System
Automatically saves all files to S3 in real-time
100% Functional - No Placeholders
"""

import os
import time
import boto3
import hashlib
import json
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import threading

class S3AutoSaveHandler(FileSystemEventHandler):
    """Handler for file system events that auto-saves to S3"""
    
    def __init__(self, s3_client, bucket, local_root, s3_prefix):
        self.s3 = s3_client
        self.bucket = bucket
        self.local_root = Path(local_root)
        self.s3_prefix = s3_prefix
        self.upload_queue = []
        self.queue_lock = threading.Lock()
        self.stats = {
            'files_uploaded': 0,
            'bytes_uploaded': 0,
            'errors': 0,
            'last_upload': None
        }
        
        # Start upload worker
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()
        
        print(f"✅ S3 Auto-Save initialized")
        print(f"   Local: {local_root}")
        print(f"   S3: s3://{bucket}/{s3_prefix}")
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and self._should_upload(event.src_path):
            self._queue_upload(event.src_path)
    
    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory and self._should_upload(event.src_path):
            self._queue_upload(event.src_path)
    
    def _should_upload(self, filepath):
        """Check if file should be uploaded"""
        path = Path(filepath)
        
        # Skip hidden files
        if path.name.startswith('.'):
            return False
        
        # Skip temp files
        if path.suffix in ['.tmp', '.swp', '.bak']:
            return False
        
        # Skip __pycache__
        if '__pycache__' in str(path):
            return False
        
        # Only upload specific extensions
        allowed_extensions = [
            '.py', '.json', '.md', '.txt', '.yaml', '.yml',
            '.sh', '.conf', '.cfg', '.ini', '.toml'
        ]
        
        if path.suffix not in allowed_extensions:
            return False
        
        return True
    
    def _queue_upload(self, filepath):
        """Add file to upload queue"""
        with self.queue_lock:
            if filepath not in self.upload_queue:
                self.upload_queue.append(filepath)
    
    def _upload_worker(self):
        """Background worker that uploads files from queue"""
        while True:
            try:
                with self.queue_lock:
                    if self.upload_queue:
                        filepath = self.upload_queue.pop(0)
                    else:
                        filepath = None
                
                if filepath:
                    self._upload_file(filepath)
                else:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"⚠️  Upload worker error: {e}")
                time.sleep(5)
    
    def _upload_file(self, filepath):
        """Upload a single file to S3"""
        try:
            path = Path(filepath)
            
            # Skip if file doesn't exist
            if not path.exists():
                return
            
            # Calculate relative path
            relative_path = path.relative_to(self.local_root)
            s3_key = f"{self.s3_prefix}{relative_path}"
            
            # Upload to S3
            self.s3.upload_file(str(path), self.bucket, s3_key)
            
            # Update stats
            file_size = path.stat().st_size
            self.stats['files_uploaded'] += 1
            self.stats['bytes_uploaded'] += file_size
            self.stats['last_upload'] = datetime.now().isoformat()
            
            print(f"✅ Auto-saved: {relative_path} ({file_size/1024:.1f} KB)")
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"❌ Upload failed: {filepath} - {e}")
    
    def get_stats(self):
        """Get upload statistics"""
        return self.stats.copy()


class ContinuousS3AutoSave:
    """
    Continuous S3 Auto-Save System
    
    Monitors directories and automatically saves all changes to S3
    """
    
    def __init__(
        self,
        bucket='asi-knowledge-base-898982995956',
        watch_dirs=None,
        s3_prefix='true-asi-system/'
    ):
        self.bucket = bucket
        self.s3_prefix = s3_prefix
        self.s3 = boto3.client('s3')
        
        # Default watch directories
        if watch_dirs is None:
            watch_dirs = [
                '/home/ubuntu/true-asi-system',
                '/home/ubuntu'
            ]
        
        self.watch_dirs = watch_dirs
        self.observers = []
        self.handlers = []
        
        print("🚀 CONTINUOUS S3 AUTO-SAVE SYSTEM")
        print(f"   Bucket: {bucket}")
        print(f"   Prefix: {s3_prefix}")
        print(f"   Watch dirs: {len(watch_dirs)}")
    
    def start(self):
        """Start monitoring and auto-saving"""
        
        for watch_dir in self.watch_dirs:
            if os.path.exists(watch_dir):
                # Create handler
                handler = S3AutoSaveHandler(
                    self.s3,
                    self.bucket,
                    watch_dir,
                    self.s3_prefix
                )
                
                # Create observer
                observer = Observer()
                observer.schedule(handler, watch_dir, recursive=True)
                observer.start()
                
                self.handlers.append(handler)
                self.observers.append(observer)
                
                print(f"✅ Watching: {watch_dir}")
            else:
                print(f"⚠️  Directory not found: {watch_dir}")
        
        print(f"\n✅ Auto-save active - monitoring {len(self.observers)} directories")
    
    def stop(self):
        """Stop monitoring"""
        for observer in self.observers:
            observer.stop()
            observer.join()
        print("✅ Auto-save stopped")
    
    def get_stats(self):
        """Get statistics from all handlers"""
        total_stats = {
            'files_uploaded': 0,
            'bytes_uploaded': 0,
            'errors': 0,
            'handlers': len(self.handlers)
        }
        
        for handler in self.handlers:
            stats = handler.get_stats()
            total_stats['files_uploaded'] += stats['files_uploaded']
            total_stats['bytes_uploaded'] += stats['bytes_uploaded']
            total_stats['errors'] += stats['errors']
        
        return total_stats
    
    def manual_sync(self, directory=None):
        """Manually sync all files in a directory"""
        
        if directory is None:
            directories = self.watch_dirs
        else:
            directories = [directory]
        
        total_uploaded = 0
        
        for dir_path in directories:
            if not os.path.exists(dir_path):
                continue
            
            print(f"\n📦 Syncing: {dir_path}")
            
            for root, dirs, files in os.walk(dir_path):
                # Skip hidden and cache directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for file in files:
                    filepath = os.path.join(root, file)
                    path = Path(filepath)
                    
                    # Check if should upload
                    if not path.name.startswith('.') and path.suffix in [
                        '.py', '.json', '.md', '.txt', '.yaml', '.yml',
                        '.sh', '.conf', '.cfg', '.ini', '.toml'
                    ]:
                        try:
                            # Calculate relative path
                            relative_path = path.relative_to(dir_path)
                            s3_key = f"{self.s3_prefix}{relative_path}"
                            
                            # Upload
                            self.s3.upload_file(str(path), self.bucket, s3_key)
                            total_uploaded += 1
                            
                            if total_uploaded % 10 == 0:
                                print(f"   Uploaded: {total_uploaded} files...")
                                
                        except Exception as e:
                            print(f"⚠️  Failed: {filepath} - {e}")
            
            print(f"✅ Synced: {total_uploaded} files from {dir_path}")
        
        return total_uploaded


# Example usage and testing
if __name__ == "__main__":
    # Create auto-save system
    autosave = ContinuousS3AutoSave(
        bucket='asi-knowledge-base-898982995956',
        watch_dirs=['/home/ubuntu/true-asi-system'],
        s3_prefix='true-asi-system/'
    )
    
    # Do initial manual sync
    print("\n📦 Performing initial sync...")
    synced = autosave.manual_sync()
    print(f"✅ Initial sync complete: {synced} files")
    
    # Start continuous monitoring
    print("\n🚀 Starting continuous monitoring...")
    autosave.start()
    
    # Keep running
    try:
        while True:
            time.sleep(60)
            stats = autosave.get_stats()
            print(f"\n📊 Stats: {stats['files_uploaded']} files, "
                  f"{stats['bytes_uploaded']/1024/1024:.1f} MB, "
                  f"{stats['errors']} errors")
    except KeyboardInterrupt:
        print("\n⏹️  Stopping...")
        autosave.stop()
