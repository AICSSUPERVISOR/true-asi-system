"""
S3 Model Registry
Phase 7: Central registry for all LLM models stored in S3

This module provides a unified interface to access all LLM models
uploaded to S3, with metadata, capabilities, and loading instructions.
"""

import boto3
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ModelInfo:
    """Information about an LLM model in S3"""
    name: str
    repo_id: str
    s3_bucket: str
    s3_prefix: str
    size_gb: float
    num_files: int
    parameters: str
    context_length: int
    license: str
    uploaded_at: str
    status: str  # 'complete', 'partial', 'pending'
    
class S3ModelRegistry:
    """
    S3 Model Registry
    
    Central registry for all LLM models stored in S3
    """
    
    def __init__(self, s3_bucket: str = "asi-knowledge-base-898982995956"):
        self.s3_bucket = s3_bucket
        self.s3 = boto3.client('s3')
        
        # Registry of all models
        self.models: Dict[str, ModelInfo] = {}
        
        # Initialize with uploaded models
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize registry with all uploaded models"""
        
        # Model 1: TinyLlama 1.1B Chat
        self.models['tinyllama-1.1b-chat'] = ModelInfo(
            name="TinyLlama 1.1B Chat v1.0",
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/tinyllama-1.1b-chat",
            size_gb=2.05,
            num_files=10,
            parameters="1.1B",
            context_length=2048,
            license="Apache 2.0",
            uploaded_at="2025-11-27T19:30:00Z",
            status="complete"
        )
        
        # Model 2: Phi-3 Mini 4K Instruct
        self.models['phi-3-mini-4k'] = ModelInfo(
            name="Phi-3 Mini 4K Instruct",
            repo_id="microsoft/Phi-3-mini-4k-instruct",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/phi-3-mini-4k-instruct",
            size_gb=7.12,
            num_files=16,
            parameters="3.8B",
            context_length=4096,
            license="MIT",
            uploaded_at="2025-11-27T19:40:00Z",
            status="complete"
        )
        
        # Model 3: Gemma 2B Instruct (partial)
        self.models['gemma-2b-it'] = ModelInfo(
            name="Gemma 2B Instruct",
            repo_id="google/gemma-2b-it",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/gemma-2b-it",
            size_gb=0.0,
            num_files=1,
            parameters="2B",
            context_length=8192,
            license="Gemma License",
            uploaded_at="2025-11-27T19:42:00Z",
            status="partial"
        )
        
        # Model 4: Llama 3.2 1B Instruct (partial)
        self.models['llama-3.2-1b'] = ModelInfo(
            name="Llama 3.2 1B Instruct",
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/llama-3.2-1b-instruct",
            size_gb=0.0,
            num_files=2,
            parameters="1B",
            context_length=128000,
            license="Llama 3.2 License",
            uploaded_at="2025-11-27T19:43:00Z",
            status="partial"
        )
    
    def list_models(self, status_filter: Optional[str] = None) -> List[ModelInfo]:
        """
        List all models in registry
        
        Args:
            status_filter: Filter by status ('complete', 'partial', 'pending')
            
        Returns:
            List of ModelInfo objects
        """
        models = list(self.models.values())
        
        if status_filter:
            models = [m for m in models if m.status == status_filter]
        
        return models
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get model info by ID
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelInfo or None
        """
        return self.models.get(model_id)
    
    def get_s3_path(self, model_id: str) -> Optional[str]:
        """
        Get S3 path for model
        
        Args:
            model_id: Model identifier
            
        Returns:
            S3 path (s3://bucket/prefix) or None
        """
        model = self.get_model(model_id)
        if model:
            return f"s3://{model.s3_bucket}/{model.s3_prefix}"
        return None
    
    def download_model_file(
        self,
        model_id: str,
        filename: str,
        local_path: str
    ) -> bool:
        """
        Download a specific file from model in S3
        
        Args:
            model_id: Model identifier
            filename: File to download
            local_path: Local destination path
            
        Returns:
            True if successful
        """
        model = self.get_model(model_id)
        if not model:
            return False
        
        try:
            s3_key = f"{model.s3_prefix}/{filename}"
            self.s3.download_file(model.s3_bucket, s3_key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False
    
    def list_model_files(self, model_id: str) -> List[str]:
        """
        List all files for a model in S3
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of filenames
        """
        model = self.get_model(model_id)
        if not model:
            return []
        
        try:
            response = self.s3.list_objects_v2(
                Bucket=model.s3_bucket,
                Prefix=model.s3_prefix + "/"
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Extract filename from full key
                    filename = obj['Key'].replace(model.s3_prefix + "/", "")
                    if filename:
                        files.append(filename)
            
            return files
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def get_registry_summary(self) -> Dict:
        """
        Get summary of all models in registry
        
        Returns:
            Summary dictionary
        """
        complete_models = [m for m in self.models.values() if m.status == 'complete']
        partial_models = [m for m in self.models.values() if m.status == 'partial']
        
        total_size = sum(m.size_gb for m in self.models.values())
        
        return {
            'total_models': len(self.models),
            'complete_models': len(complete_models),
            'partial_models': len(partial_models),
            'total_size_gb': total_size,
            'models': {
                model_id: asdict(model)
                for model_id, model in self.models.items()
            }
        }
    
    def save_registry_to_s3(self):
        """Save registry to S3 as JSON"""
        summary = self.get_registry_summary()
        summary['updated_at'] = datetime.utcnow().isoformat()
        
        self.s3.put_object(
            Bucket=self.s3_bucket,
            Key="true-asi-system/models/model_registry.json",
            Body=json.dumps(summary, indent=2),
            ContentType='application/json'
        )
        
        print(f"✅ Registry saved to S3: s3://{self.s3_bucket}/true-asi-system/models/model_registry.json")


# Example usage
if __name__ == "__main__":
    # Initialize registry
    registry = S3ModelRegistry()
    
    # Get summary
    summary = registry.get_registry_summary()
    
    print("📊 S3 MODEL REGISTRY")
    print("=" * 70)
    print(f"Total Models: {summary['total_models']}")
    print(f"Complete Models: {summary['complete_models']}")
    print(f"Partial Models: {summary['partial_models']}")
    print(f"Total Size: {summary['total_size_gb']:.2f} GB")
    print("=" * 70)
    print("")
    
    # List complete models
    print("✅ COMPLETE MODELS:")
    for model in registry.list_models(status_filter='complete'):
        print(f"  • {model.name}")
        print(f"    ID: {model.name.lower().replace(' ', '-')}")
        print(f"    Parameters: {model.parameters}")
        print(f"    Size: {model.size_gb:.2f} GB")
        print(f"    Context: {model.context_length:,} tokens")
        print(f"    S3: s3://{model.s3_bucket}/{model.s3_prefix}")
        print("")
    
    # Save to S3
    registry.save_registry_to_s3()
    
    print("✅ PHASE 7 COMPLETE: S3 Model Registry Created")
