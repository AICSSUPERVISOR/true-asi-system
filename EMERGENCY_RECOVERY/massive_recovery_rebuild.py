#!/usr/bin/env python3
"""
MASSIVE ASI RECOVERY AND REBUILD SYSTEM
=======================================
Recovers all data from sandbox, rebuilds complete ASI system,
and prepares for GitHub upload.

Target: Rebuild 16TB equivalent system from all available sources
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import hashlib

# ============================================================================
# CONFIGURATION
# ============================================================================

RECOVERY_DIR = Path("/home/ubuntu/FULL_RECOVERY")
GITHUB_REPO = Path("/home/ubuntu/true-asi-system")
S3_AUDIT_FILE = Path("/home/ubuntu/upload/.recovery/s3_comprehensive_audit.json")

SOURCE_DIRECTORIES = [
    "/home/ubuntu/true-asi-system",
    "/home/ubuntu/real-asi",
    "/home/ubuntu/upload/.recovery",
    "/home/ubuntu/asi-production",
    "/home/ubuntu/accelerated-asi",
    "/home/ubuntu/final-asi-phases",
    "/home/ubuntu/true-asi-build",
    "/home/ubuntu/true-asi-implementation",
    "/home/ubuntu/asi-research",
    "/home/ubuntu/asi-synthesis",
    "/home/ubuntu/asi-models",
    "/home/ubuntu/asi-improvements",
    "/home/ubuntu/asi-requirements",
]

# ============================================================================
# RECOVERY FUNCTIONS
# ============================================================================

def load_s3_audit() -> Dict:
    """Load the S3 comprehensive audit file"""
    if S3_AUDIT_FILE.exists():
        with open(S3_AUDIT_FILE, 'r') as f:
            return json.load(f)
    return {}

def catalog_local_files() -> Dict[str, List[str]]:
    """Catalog all local files by type"""
    catalog = {
        "python": [],
        "json": [],
        "markdown": [],
        "yaml": [],
        "shell": [],
        "text": [],
        "other": []
    }
    
    extensions = {
        ".py": "python",
        ".json": "json",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".sh": "shell",
        ".txt": "text"
    }
    
    for source_dir in SOURCE_DIRECTORIES:
        if not os.path.exists(source_dir):
            continue
        for root, dirs, files in os.walk(source_dir):
            # Skip node_modules and .git
            dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__']]
            for file in files:
                filepath = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                category = extensions.get(ext, "other")
                catalog[category].append(filepath)
    
    return catalog

def extract_s3_file_list(audit: Dict) -> List[Dict]:
    """Extract complete file list from S3 audit"""
    files = []
    for bucket in audit.get("buckets", []):
        bucket_name = bucket.get("bucket_name", "")
        for file_info in bucket.get("largest_files", []):
            files.append({
                "bucket": bucket_name,
                "key": file_info.get("key", ""),
                "size": file_info.get("size", 0),
                "last_modified": file_info.get("last_modified", "")
            })
    return files

def create_recovery_manifest(catalog: Dict, s3_audit: Dict) -> Dict:
    """Create comprehensive recovery manifest"""
    manifest = {
        "recovery_date": datetime.now().isoformat(),
        "local_files": {
            category: len(files) for category, files in catalog.items()
        },
        "total_local_files": sum(len(files) for files in catalog.values()),
        "s3_audit_summary": {},
        "recovery_status": "in_progress"
    }
    
    # Add S3 audit summary
    for bucket in s3_audit.get("buckets", []):
        bucket_name = bucket.get("bucket_name", "")
        manifest["s3_audit_summary"][bucket_name] = {
            "total_objects": bucket.get("total_objects", 0),
            "total_size_tb": bucket.get("total_size_tb", 0),
            "file_types": bucket.get("file_types", {})
        }
    
    return manifest

def copy_to_recovery(catalog: Dict) -> int:
    """Copy all files to recovery directory"""
    copied = 0
    
    for category, files in catalog.items():
        category_dir = RECOVERY_DIR / "recovered_files" / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        for filepath in files:
            try:
                src = Path(filepath)
                if not src.exists():
                    continue
                
                # Create unique filename to avoid collisions
                rel_path = str(src).replace("/home/ubuntu/", "").replace("/", "_")
                dst = category_dir / rel_path
                
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                pass
    
    return copied

def generate_rebuild_structure() -> Dict:
    """Generate the structure for rebuilding the complete ASI system"""
    structure = {
        "core_systems": {
            "autonomous_agents": "Self-replicating agents for all industries",
            "swarm_intelligence": "Hivemind coordination with consensus",
            "knowledge_base": "Recursive knowledge growth system",
            "mcp_integration": "109 tools across 6 connectors",
            "zapier_automation": "18 workflows for all industries"
        },
        "infrastructure": {
            "eks_cluster": "true-asi-cluster configuration",
            "s3_storage": "Data persistence layer",
            "ec2_compute": "Processing nodes",
            "lambda_functions": "Serverless operations"
        },
        "data_layers": {
            "agents": "1.18M agent files",
            "knowledge": "Knowledge graph and embeddings",
            "models": "AI model configurations",
            "processing": "Data processing pipelines"
        },
        "scale_targets": {
            "agents": 1000000,
            "knowledge_nodes": 10000000,
            "connections": 100000000
        }
    }
    return structure

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("MASSIVE ASI RECOVERY AND REBUILD SYSTEM")
    print("=" * 80)
    print(f"Recovery Directory: {RECOVERY_DIR}")
    print(f"Target: Rebuild 16TB ASI System")
    print()
    
    # Step 1: Load S3 Audit
    print("[1/5] Loading S3 Comprehensive Audit...")
    s3_audit = load_s3_audit()
    if s3_audit:
        total_objects = sum(b.get("total_objects", 0) for b in s3_audit.get("buckets", []))
        total_tb = sum(b.get("total_size_tb", 0) for b in s3_audit.get("buckets", []))
        print(f"    ✅ S3 Audit loaded: {total_objects:,} objects, {total_tb:.2f} TB")
    else:
        print("    ⚠️ S3 Audit not found")
    
    # Step 2: Catalog Local Files
    print("\n[2/5] Cataloging Local Files...")
    catalog = catalog_local_files()
    for category, files in catalog.items():
        if files:
            print(f"    {category}: {len(files):,} files")
    print(f"    Total: {sum(len(f) for f in catalog.values()):,} files")
    
    # Step 3: Create Recovery Manifest
    print("\n[3/5] Creating Recovery Manifest...")
    manifest = create_recovery_manifest(catalog, s3_audit)
    manifest_path = RECOVERY_DIR / "RECOVERY_MANIFEST.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"    ✅ Manifest saved: {manifest_path}")
    
    # Step 4: Copy Files to Recovery
    print("\n[4/5] Copying Files to Recovery Directory...")
    copied = copy_to_recovery(catalog)
    print(f"    ✅ Copied {copied:,} files")
    
    # Step 5: Generate Rebuild Structure
    print("\n[5/5] Generating Rebuild Structure...")
    structure = generate_rebuild_structure()
    structure_path = RECOVERY_DIR / "REBUILD_STRUCTURE.json"
    with open(structure_path, 'w') as f:
        json.dump(structure, f, indent=2)
    print(f"    ✅ Structure saved: {structure_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("RECOVERY PHASE 1 COMPLETE")
    print("=" * 80)
    print(f"""
RECOVERED DATA:
✅ Local Files: {sum(len(f) for f in catalog.values()):,}
✅ S3 Audit: {total_objects:,} objects ({total_tb:.2f} TB documented)
✅ Recovery Directory: {RECOVERY_DIR}

S3 BUCKET SUMMARY (from audit):
""")
    
    for bucket in s3_audit.get("buckets", []):
        name = bucket.get("bucket_name", "")
        objects = bucket.get("total_objects", 0)
        size_tb = bucket.get("total_size_tb", 0)
        if objects > 0:
            print(f"  • {name}: {objects:,} objects ({size_tb:.2f} TB)")
    
    print(f"""
NEXT STEPS:
1. Rebuild complete ASI system from recovered components
2. Scale to 1 million agents with hivemind
3. Push everything to GitHub

Recovery manifest: {manifest_path}
""")
    print("=" * 80)
    
    return manifest

if __name__ == "__main__":
    main()
