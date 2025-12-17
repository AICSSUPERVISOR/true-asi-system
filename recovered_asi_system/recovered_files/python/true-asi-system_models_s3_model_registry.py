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
        """Initialize registry with all 18 uploaded models"""
        
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
        
        # Model 2: Phi-2
        self.models['phi-2'] = ModelInfo(
            name="Phi-2",
            repo_id="microsoft/phi-2",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/phi-2",
            size_gb=5.18,
            num_files=53,
            parameters="2.7B",
            context_length=2048,
            license="MIT",
            uploaded_at="2025-11-27T19:35:00Z",
            status="complete"
        )
        
        # Model 3: Phi-1.5
        self.models['phi-1_5'] = ModelInfo(
            name="Phi-1.5",
            repo_id="microsoft/phi-1_5",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/phi-1_5",
            size_gb=2.64,
            num_files=50,
            parameters="1.3B",
            context_length=2048,
            license="MIT",
            uploaded_at="2025-11-27T19:36:00Z",
            status="complete"
        )
        
        # Model 4: Phi-3 Mini 4K Instruct
        self.models['phi-3-mini-4k-instruct'] = ModelInfo(
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
        
        # Model 5: Qwen2 0.5B
        self.models['qwen-qwen2-0.5b'] = ModelInfo(
            name="Qwen2 0.5B",
            repo_id="Qwen/Qwen2-0.5B",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/qwen-qwen2-0.5b",
            size_gb=0.93,
            num_files=32,
            parameters="0.5B",
            context_length=32768,
            license="Apache 2.0",
            uploaded_at="2025-11-27T19:45:00Z",
            status="complete"
        )
        
        # Model 6: Qwen2 1.5B
        self.models['qwen-qwen2-1.5b'] = ModelInfo(
            name="Qwen2 1.5B",
            repo_id="Qwen/Qwen2-1.5B",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/qwen-qwen2-1.5b",
            size_gb=2.89,
            num_files=32,
            parameters="1.5B",
            context_length=32768,
            license="Apache 2.0",
            uploaded_at="2025-11-27T19:46:00Z",
            status="complete"
        )
        
        # Model 7: StableLM 2 1.6B
        self.models['stabilityai-stablelm-2-1_6b'] = ModelInfo(
            name="StableLM 2 1.6B",
            repo_id="stabilityai/stablelm-2-1_6b",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/stabilityai-stablelm-2-1_6b",
            size_gb=3.07,
            num_files=41,
            parameters="1.6B",
            context_length=4096,
            license="Apache 2.0",
            uploaded_at="2025-11-27T19:47:00Z",
            status="complete"
        )
        
        # Model 8: StableLM Zephyr 3B
        self.models['stabilityai-stablelm-zephyr-3b'] = ModelInfo(
            name="StableLM Zephyr 3B",
            repo_id="stabilityai/stablelm-zephyr-3b",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/stabilityai-stablelm-zephyr-3b",
            size_gb=5.21,
            num_files=35,
            parameters="3B",
            context_length=4096,
            license="Apache 2.0",
            uploaded_at="2025-11-27T19:48:00Z",
            status="complete"
        )
        
        # Model 9: CodeGen 2B Mono
        self.models['salesforce-codegen-2b-mono'] = ModelInfo(
            name="CodeGen 2B Mono",
            repo_id="Salesforce/codegen-2B-mono",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/salesforce-codegen-2b-mono",
            size_gb=5.31,
            num_files=32,
            parameters="2B",
            context_length=2048,
            license="Apache 2.0",
            uploaded_at="2025-11-27T19:50:00Z",
            status="complete"
        )
        
        # Model 10: CodeGen 2.5 7B Mono
        self.models['salesforce-codegen25-7b-mono'] = ModelInfo(
            name="CodeGen 2.5 7B Mono",
            repo_id="Salesforce/codegen25-7b-mono",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/salesforce-codegen25-7b-mono",
            size_gb=25.69,
            num_files=32,
            parameters="7B",
            context_length=2048,
            license="Apache 2.0",
            uploaded_at="2025-11-27T19:55:00Z",
            status="complete"
        )
        
        # Model 11: Llemma 7B
        self.models['eleutherai-llemma_7b'] = ModelInfo(
            name="Llemma 7B",
            repo_id="EleutherAI/llemma_7b",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/eleutherai-llemma_7b",
            size_gb=25.11,
            num_files=38,
            parameters="7B",
            context_length=4096,
            license="Apache 2.0",
            uploaded_at="2025-11-27T20:00:00Z",
            status="complete"
        )
        
        # Model 12: Replit Code v1.5 3B
        self.models['replit-replit-code-v1_5-3b'] = ModelInfo(
            name="Replit Code v1.5 3B",
            repo_id="replit/replit-code-v1_5-3b",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/replit-replit-code-v1_5-3b",
            size_gb=6.19,
            num_files=65,
            parameters="3B",
            context_length=4096,
            license="Apache 2.0",
            uploaded_at="2025-11-27T20:05:00Z",
            status="complete"
        )
        
        # Model 13: InCoder 1B
        self.models['facebook-incoder-1b'] = ModelInfo(
            name="InCoder 1B",
            repo_id="facebook/incoder-1B",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/facebook-incoder-1b",
            size_gb=2.45,
            num_files=26,
            parameters="1B",
            context_length=2048,
            license="Apache 2.0",
            uploaded_at="2025-11-27T20:10:00Z",
            status="complete"
        )
        
        # Model 14: CodeBERT
        self.models['codebert'] = ModelInfo(
            name="CodeBERT",
            repo_id="microsoft/codebert-base",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/codebert",
            size_gb=1.86,
            num_files=34,
            parameters="125M",
            context_length=512,
            license="MIT",
            uploaded_at="2025-11-27T20:12:00Z",
            status="complete"
        )
        
        # Model 15: GraphCodeBERT
        self.models['graphcodebert'] = ModelInfo(
            name="GraphCodeBERT",
            repo_id="microsoft/graphcodebert-base",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/graphcodebert",
            size_gb=1.54,
            num_files=31,
            parameters="125M",
            context_length=512,
            license="MIT",
            uploaded_at="2025-11-27T20:13:00Z",
            status="complete"
        )
        
        # Model 16: CodeRL 770M
        self.models['coderl-770m'] = ModelInfo(
            name="CodeRL 770M",
            repo_id="salesforce/coderl-770m",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/coderl-770m",
            size_gb=0.75,
            num_files=31,
            parameters="770M",
            context_length=2048,
            license="Apache 2.0",
            uploaded_at="2025-11-27T20:14:00Z",
            status="complete"
        )
        
        # Model 17: PyCodeGPT 110M
        self.models['pycodegpt-110m'] = ModelInfo(
            name="PyCodeGPT 110M",
            repo_id="microsoft/pycodegpt-110m",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/pycodegpt-110m",
            size_gb=1.40,
            num_files=34,
            parameters="110M",
            context_length=1024,
            license="MIT",
            uploaded_at="2025-11-27T20:15:00Z",
            status="complete"
        )
        
        # Model 18: UniXcoder
        self.models['unixcoder'] = ModelInfo(
            name="UniXcoder",
            repo_id="microsoft/unixcoder-base",
            s3_bucket=self.s3_bucket,
            s3_prefix="true-asi-system/models/unixcoder",
            size_gb=0.47,
            num_files=25,
            parameters="125M",
            context_length=1024,
            license="MIT",
            uploaded_at="2025-11-27T20:16:00Z",
            status="complete"
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
