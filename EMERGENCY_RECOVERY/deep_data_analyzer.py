#!/usr/bin/env python3.11
"""
Deep Data Analyzer for 10.17 TB AWS S3 Knowledge Base
Categorizes and analyzes all data for True ASI integration
"""

import boto3
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

class DeepDataAnalyzer:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket = 'asi-knowledge-base-898982995956'
        
        self.categories = {
            'models': defaultdict(int),
            'code': defaultdict(int),
            'agents': defaultdict(int),
            'documentation': defaultdict(int),
            'data': defaultdict(int),
            'infrastructure': defaultdict(int),
            'reports': defaultdict(int),
            'backups': defaultdict(int),
            'repositories': defaultdict(int),
            'training': defaultdict(int)
        }
        
        self.analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_size_tb": 10.1729,
            "total_objects": 1183525,
            "categories": {},
            "model_weights": [],
            "agent_systems": [],
            "code_repositories": [],
            "key_findings": []
        }
    
    def categorize_path(self, path: str) -> str:
        """Categorize S3 object by path"""
        path_lower = path.lower()
        
        # Model weights
        if any(x in path_lower for x in ['model', 'pytorch', 'safetensors', 'weights', 'checkpoint']):
            return 'models'
        
        # Agent systems
        if any(x in path_lower for x in ['agent', 'swarm', 'coordination']):
            return 'agents'
        
        # Code repositories
        if any(x in path_lower for x in ['repo', 'github', 'code/', 'src/']):
            return 'repositories'
        
        # Documentation
        if any(x in path_lower for x in ['doc', 'readme', 'guide', '.md']):
            return 'documentation'
        
        # Infrastructure
        if any(x in path_lower for x in ['infrastructure', 'deployment', 'terraform', 'ec2', 'kubernetes']):
            return 'infrastructure'
        
        # Reports and audits
        if any(x in path_lower for x in ['report', 'audit', 'status', 'summary']):
            return 'reports'
        
        # Backups
        if any(x in path_lower for x in ['backup', 'archive', '.tar', '.zip']):
            return 'backups'
        
        # Training data
        if any(x in path_lower for x in ['training', 'dataset', 'data/']):
            return 'training'
        
        # Default to data
        return 'data'
    
    def analyze_top_level_structure(self):
        """Analyze top-level folder structure"""
        print("Analyzing top-level S3 structure...")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Delimiter='/'
            )
            
            folders = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    folder_name = prefix['Prefix'].rstrip('/')
                    folders.append(folder_name)
            
            print(f"Found {len(folders)} top-level folders")
            
            return folders
        
        except Exception as e:
            print(f"Error analyzing structure: {e}")
            return []
    
    def sample_analysis(self, max_objects=10000):
        """Perform sample analysis of data"""
        print(f"\nPerforming sample analysis of {max_objects} objects...")
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.bucket,
                PaginationConfig={'MaxItems': max_objects}
            )
            
            count = 0
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        size = obj.get('Size', 0)
                        
                        # Categorize
                        category = self.categorize_path(key)
                        self.categories[category][key] = size
                        
                        count += 1
                        
                        if count % 1000 == 0:
                            print(f"  Analyzed {count} objects...")
            
            print(f"✅ Sample analysis complete: {count} objects analyzed")
            
        except Exception as e:
            print(f"Error in sample analysis: {e}")
    
    def identify_key_components(self):
        """Identify key system components"""
        print("\nIdentifying key system components...")
        
        key_findings = []
        
        # Analyze models
        if self.categories['models']:
            model_count = len(self.categories['models'])
            model_size = sum(self.categories['models'].values()) / (1024**3)
            key_findings.append(f"Found {model_count} model files ({model_size:.2f} GB)")
            
            # Identify largest models
            largest_models = sorted(
                self.categories['models'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            self.analysis['model_weights'] = [
                {
                    'path': path,
                    'size_gb': round(size / (1024**3), 2)
                }
                for path, size in largest_models
            ]
        
        # Analyze agents
        if self.categories['agents']:
            agent_count = len(self.categories['agents'])
            key_findings.append(f"Found {agent_count} agent-related files")
            
            self.analysis['agent_systems'] = list(self.categories['agents'].keys())[:20]
        
        # Analyze repositories
        if self.categories['repositories']:
            repo_count = len(self.categories['repositories'])
            key_findings.append(f"Found {repo_count} repository files")
            
            self.analysis['code_repositories'] = list(self.categories['repositories'].keys())[:20]
        
        self.analysis['key_findings'] = key_findings
        
        for finding in key_findings:
            print(f"  • {finding}")
    
    def generate_category_summary(self):
        """Generate summary by category"""
        print("\nGenerating category summary...")
        
        for category, items in self.categories.items():
            if items:
                count = len(items)
                total_size = sum(items.values())
                size_gb = total_size / (1024**3)
                
                self.analysis['categories'][category] = {
                    'count': count,
                    'size_bytes': total_size,
                    'size_gb': round(size_gb, 2)
                }
                
                print(f"  {category}: {count} items, {size_gb:.2f} GB")
    
    def save_analysis(self, filepath: str):
        """Save analysis results"""
        with open(filepath, 'w') as f:
            json.dump(self.analysis, f, indent=2)
        
        print(f"\n✅ Analysis saved to: {filepath}")
    
    def run_analysis(self):
        """Run complete analysis"""
        print("="*80)
        print("DEEP DATA ANALYSIS - 10.17 TB AWS S3 KNOWLEDGE BASE")
        print("="*80)
        
        # Analyze structure
        folders = self.analyze_top_level_structure()
        self.analysis['top_level_folders'] = folders
        
        # Sample analysis
        self.sample_analysis(max_objects=10000)
        
        # Identify components
        self.identify_key_components()
        
        # Generate summary
        self.generate_category_summary()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)

if __name__ == "__main__":
    analyzer = DeepDataAnalyzer()
    analyzer.run_analysis()
    analyzer.save_analysis("/home/ubuntu/true-asi-build/deep_data_analysis.json")
