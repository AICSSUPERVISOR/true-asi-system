#!/usr/bin/env python3.11
"""
Comprehensive AWS S3 Audit - Locate 6TB+ Data
"""

import boto3
import json
from datetime import datetime
from typing import Dict, List, Any

class ComprehensiveS3Audit:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.audit_results = {
            "audit_date": datetime.now().isoformat(),
            "buckets": [],
            "total_size_bytes": 0,
            "total_size_gb": 0,
            "total_size_tb": 0,
            "total_objects": 0
        }
    
    def get_bucket_size(self, bucket_name: str) -> Dict[str, Any]:
        """Get comprehensive bucket statistics"""
        print(f"\nAuditing bucket: {bucket_name}")
        
        try:
            # Get bucket location
            location = self.s3_client.get_bucket_location(Bucket=bucket_name)
            region = location.get('LocationConstraint', 'us-east-1')
            
            # List all objects with pagination
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name)
            
            total_size = 0
            object_count = 0
            file_types = {}
            largest_files = []
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        size = obj.get('Size', 0)
                        total_size += size
                        object_count += 1
                        
                        # Track file types
                        key = obj['Key']
                        ext = key.split('.')[-1] if '.' in key else 'no_extension'
                        file_types[ext] = file_types.get(ext, 0) + 1
                        
                        # Track largest files
                        largest_files.append({
                            'key': key,
                            'size': size,
                            'size_mb': round(size / (1024*1024), 2),
                            'last_modified': obj['LastModified'].isoformat()
                        })
                
                # Progress indicator
                if object_count % 1000 == 0 and object_count > 0:
                    print(f"  Processed {object_count} objects, {round(total_size / (1024**3), 2)} GB so far...")
            
            # Sort largest files
            largest_files.sort(key=lambda x: x['size'], reverse=True)
            top_10_largest = largest_files[:10]
            
            size_gb = total_size / (1024**3)
            size_tb = total_size / (1024**4)
            
            bucket_info = {
                "bucket_name": bucket_name,
                "region": region,
                "total_objects": object_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024**2), 2),
                "total_size_gb": round(size_gb, 2),
                "total_size_tb": round(size_tb, 4),
                "file_types": file_types,
                "largest_files": top_10_largest
            }
            
            print(f"  ✅ {bucket_name}:")
            print(f"     Objects: {object_count:,}")
            print(f"     Size: {round(size_gb, 2)} GB ({round(size_tb, 4)} TB)")
            
            return bucket_info
        
        except Exception as e:
            print(f"  ❌ Error auditing {bucket_name}: {e}")
            return {
                "bucket_name": bucket_name,
                "error": str(e),
                "total_objects": 0,
                "total_size_bytes": 0,
                "total_size_gb": 0,
                "total_size_tb": 0
            }
    
    def audit_all_buckets(self):
        """Audit all S3 buckets in the account"""
        print("="*80)
        print("COMPREHENSIVE AWS S3 AUDIT - LOCATING 6TB+ DATA")
        print("="*80)
        
        # List all buckets
        try:
            response = self.s3_client.list_buckets()
            buckets = response.get('Buckets', [])
            
            print(f"\nFound {len(buckets)} buckets in AWS account")
            print("-"*80)
            
            for bucket in buckets:
                bucket_name = bucket['Name']
                bucket_info = self.get_bucket_size(bucket_name)
                self.audit_results['buckets'].append(bucket_info)
                
                self.audit_results['total_size_bytes'] += bucket_info.get('total_size_bytes', 0)
                self.audit_results['total_objects'] += bucket_info.get('total_objects', 0)
        
        except Exception as e:
            print(f"Error listing buckets: {e}")
        
        # Calculate totals
        self.audit_results['total_size_gb'] = round(
            self.audit_results['total_size_bytes'] / (1024**3), 2
        )
        self.audit_results['total_size_tb'] = round(
            self.audit_results['total_size_bytes'] / (1024**4), 4
        )
        
        # Print summary
        print("\n" + "="*80)
        print("AUDIT SUMMARY")
        print("="*80)
        print(f"Total Buckets: {len(self.audit_results['buckets'])}")
        print(f"Total Objects: {self.audit_results['total_objects']:,}")
        print(f"Total Size: {self.audit_results['total_size_gb']:,.2f} GB")
        print(f"Total Size: {self.audit_results['total_size_tb']:.4f} TB")
        print("="*80)
        
        # Identify largest buckets
        print("\nLARGEST BUCKETS:")
        sorted_buckets = sorted(
            self.audit_results['buckets'],
            key=lambda x: x.get('total_size_bytes', 0),
            reverse=True
        )
        
        for i, bucket in enumerate(sorted_buckets[:5], 1):
            print(f"{i}. {bucket['bucket_name']}: {bucket.get('total_size_gb', 0)} GB ({bucket.get('total_size_tb', 0)} TB)")
        
        return self.audit_results
    
    def save_audit_report(self, filepath: str):
        """Save audit results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.audit_results, f, indent=2)
        
        print(f"\n✅ Audit report saved to: {filepath}")
    
    def generate_markdown_report(self, filepath: str):
        """Generate detailed markdown report"""
        report = f"""# AWS S3 COMPREHENSIVE AUDIT REPORT

**Date**: {self.audit_results['audit_date']}  
**Purpose**: Locate and verify 6TB+ ASI data storage

---

## EXECUTIVE SUMMARY

**Total Buckets**: {len(self.audit_results['buckets'])}  
**Total Objects**: {self.audit_results['total_objects']:,}  
**Total Size**: {self.audit_results['total_size_gb']:,.2f} GB ({self.audit_results['total_size_tb']:.4f} TB)

---

## BUCKET-BY-BUCKET ANALYSIS

"""
        
        sorted_buckets = sorted(
            self.audit_results['buckets'],
            key=lambda x: x.get('total_size_bytes', 0),
            reverse=True
        )
        
        for i, bucket in enumerate(sorted_buckets, 1):
            report += f"""
### {i}. {bucket['bucket_name']}

- **Region**: {bucket.get('region', 'N/A')}
- **Objects**: {bucket.get('total_objects', 0):,}
- **Size**: {bucket.get('total_size_gb', 0):,.2f} GB ({bucket.get('total_size_tb', 0):.4f} TB)
- **Size (MB)**: {bucket.get('total_size_mb', 0):,.2f} MB

"""
            
            if 'error' in bucket:
                report += f"**Error**: {bucket['error']}\n\n"
            else:
                # File types
                if bucket.get('file_types'):
                    report += "**File Types**:\n"
                    for ext, count in sorted(bucket['file_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
                        report += f"- `.{ext}`: {count:,} files\n"
                    report += "\n"
                
                # Largest files
                if bucket.get('largest_files'):
                    report += "**Largest Files**:\n"
                    for j, file in enumerate(bucket['largest_files'][:5], 1):
                        report += f"{j}. `{file['key']}` - {file['size_mb']} MB\n"
                    report += "\n"
        
        report += f"""
---

## RECOMMENDATIONS

"""
        
        if self.audit_results['total_size_tb'] >= 6.0:
            report += f"""
✅ **6TB+ DATA LOCATED**: Found {self.audit_results['total_size_tb']:.4f} TB of data across {len(self.audit_results['buckets'])} buckets.

**Next Steps**:
1. Verify data integrity and accessibility
2. Catalog all data by type and purpose
3. Integrate into True ASI system architecture
4. Create unified access layer
"""
        else:
            report += f"""
⚠️ **DATA DISCREPANCY**: Currently found {self.audit_results['total_size_tb']:.4f} TB of data.

**Possible Reasons**:
1. Data may be in different AWS regions
2. Data may be in EC2 instances or EBS volumes
3. Data may be in other AWS services (RDS, DynamoDB, etc.)
4. Need to check AWS account for additional resources

**Next Steps**:
1. Check all AWS regions
2. Audit EC2 instances and EBS volumes
3. Check RDS databases
4. Check DynamoDB tables
5. Review AWS billing for storage usage
"""
        
        report += "\n---\n\nEND OF AUDIT REPORT\n"
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"✅ Markdown report saved to: {filepath}")

if __name__ == "__main__":
    auditor = ComprehensiveS3Audit()
    results = auditor.audit_all_buckets()
    
    # Save reports
    auditor.save_audit_report("/home/ubuntu/true-asi-build/s3_comprehensive_audit.json")
    auditor.generate_markdown_report("/home/ubuntu/true-asi-build/S3_COMPREHENSIVE_AUDIT_REPORT.md")
    
    print("\n✅ Comprehensive S3 audit complete!")
