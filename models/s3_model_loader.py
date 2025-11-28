"""
S3 Model Loader and Inference Engine
Phases 8-9: Stream models from S3 and run inference

This module provides infrastructure to:
1. Stream model files from S3 to local cache
2. Load models into memory
3. Run inference
4. Clean up cache when done
"""

import boto3
import os
import shutil
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not available")

from s3_model_registry import S3ModelRegistry, ModelInfo


class S3ModelLoader:
    """
    S3 Model Loader
    
    Stream models from S3 and load for inference
    """
    
    def __init__(
        self,
        cache_dir: str = "/tmp/model_cache",
        s3_bucket: str = "asi-knowledge-base-898982995956"
    ):
        self.cache_dir = Path(cache_dir)
        self.s3_bucket = s3_bucket
        self.s3 = boto3.client('s3')
        self.registry = S3ModelRegistry(s3_bucket)
        
        # Currently loaded model
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_id = None
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_model_from_s3(
        self,
        model_id: str,
        force_download: bool = False
    ) -> Optional[Path]:
        """
        Download model from S3 to local cache
        
        Args:
            model_id: Model identifier
            force_download: Force re-download even if cached
            
        Returns:
            Path to local model directory or None
        """
        # Get model info
        model_info = self.registry.get_model(model_id)
        if not model_info:
            print(f"❌ Model not found: {model_id}")
            return None
        
        # Local cache path
        local_model_dir = self.cache_dir / model_id
        
        # Check if already cached
        if local_model_dir.exists() and not force_download:
            print(f"✅ Model already cached: {local_model_dir}")
            return local_model_dir
        
        # Create directory
        local_model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Downloading {model_info.name} from S3...")
        print(f"   S3: s3://{model_info.s3_bucket}/{model_info.s3_prefix}")
        print(f"   Local: {local_model_dir}")
        
        try:
            # List all files in S3
            response = self.s3.list_objects_v2(
                Bucket=model_info.s3_bucket,
                Prefix=model_info.s3_prefix + "/"
            )
            
            if 'Contents' not in response:
                print(f"❌ No files found in S3")
                return None
            
            # Download each file
            files_downloaded = 0
            total_size = 0
            
            for obj in response['Contents']:
                s3_key = obj['Key']
                filename = s3_key.replace(model_info.s3_prefix + "/", "")
                
                if not filename or filename == "model_manifest.json":
                    continue
                
                local_file = local_model_dir / filename
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Download
                print(f"   Downloading: {filename}")
                self.s3.download_file(
                    model_info.s3_bucket,
                    s3_key,
                    str(local_file)
                )
                
                files_downloaded += 1
                total_size += obj['Size']
            
            print(f"✅ Downloaded {files_downloaded} files ({total_size / (1024**3):.2f} GB)")
            return local_model_dir
            
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            # Clean up partial download
            if local_model_dir.exists():
                shutil.rmtree(local_model_dir)
            return None
    
    def load_model(
        self,
        model_id: str,
        device: str = "cpu",
        load_in_8bit: bool = False
    ) -> bool:
        """
        Load model into memory for inference
        
        Args:
            model_id: Model identifier
            device: Device to load on ('cpu', 'cuda')
            load_in_8bit: Use 8-bit quantization
            
        Returns:
            True if successful
        """
        if not TRANSFORMERS_AVAILABLE:
            print("❌ transformers library not available")
            return False
        
        # Download model if needed
        local_model_dir = self.download_model_from_s3(model_id)
        if not local_model_dir:
            return False
        
        print(f"🔄 Loading model into memory...")
        
        try:
            # Load tokenizer
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                str(local_model_dir),
                local_files_only=True
            )
            print(f"✅ Tokenizer loaded")
            
            # Load model
            load_kwargs = {
                'local_files_only': True,
                'low_cpu_mem_usage': True
            }
            
            if load_in_8bit and device == "cuda":
                load_kwargs['load_in_8bit'] = True
            
            self.current_model = AutoModelForCausalLM.from_pretrained(
                str(local_model_dir),
                **load_kwargs
            )
            
            # Move to device
            if not load_in_8bit:
                self.current_model = self.current_model.to(device)
            
            self.current_model_id = model_id
            
            print(f"✅ Model loaded on {device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Optional[str]:
        """
        Generate text from loaded model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text or None
        """
        if not self.current_model or not self.current_tokenizer:
            print("❌ No model loaded")
            return None
        
        try:
            # Tokenize input
            inputs = self.current_tokenizer(
                prompt,
                return_tensors="pt"
            )
            
            # Move to same device as model
            device = next(self.current_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            
            # Decode
            generated_text = self.current_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            return generated_text
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None
    
    def unload_model(self):
        """Unload current model from memory"""
        if self.current_model:
            del self.current_model
            self.current_model = None
        
        if self.current_tokenizer:
            del self.current_tokenizer
            self.current_tokenizer = None
        
        self.current_model_id = None
        
        # Clear CUDA cache if available
        if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Model unloaded from memory")
    
    def clear_cache(self):
        """Clear all cached models"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Cache cleared")


class ModelInferenceEngine:
    """
    Model Inference Engine
    
    High-level interface for running inference with S3 models
    """
    
    def __init__(self):
        self.loader = S3ModelLoader()
        self.registry = S3ModelRegistry()
    
    def list_available_models(self) -> List[str]:
        """List all available models"""
        complete_models = self.registry.list_models(status_filter='complete')
        return [
            model_id
            for model_id, model in self.registry.models.items()
            if model.status == 'complete'
        ]
    
    def run_inference(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 100
    ) -> Optional[str]:
        """
        Run inference with specified model
        
        Args:
            model_id: Model to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or None
        """
        # Load model if not already loaded
        if self.loader.current_model_id != model_id:
            self.loader.unload_model()
            if not self.loader.load_model(model_id):
                return None
        
        # Generate
        return self.loader.generate(prompt, max_new_tokens=max_tokens)
    
    def benchmark_model(
        self,
        model_id: str,
        test_prompts: List[str]
    ) -> Dict[str, Any]:
        """
        Benchmark model performance
        
        Args:
            model_id: Model to benchmark
            test_prompts: List of test prompts
            
        Returns:
            Benchmark results
        """
        import time
        
        results = {
            'model_id': model_id,
            'num_prompts': len(test_prompts),
            'results': []
        }
        
        for prompt in test_prompts:
            start_time = time.time()
            output = self.run_inference(model_id, prompt, max_tokens=50)
            elapsed = time.time() - start_time
            
            results['results'].append({
                'prompt': prompt,
                'output': output,
                'time_seconds': elapsed
            })
        
        avg_time = sum(r['time_seconds'] for r in results['results']) / len(results['results'])
        results['average_time_seconds'] = avg_time
        
        return results


# Example usage
if __name__ == "__main__":
    print("🚀 S3 MODEL LOADER & INFERENCE ENGINE")
    print("=" * 70)
    
    # Initialize engine
    engine = ModelInferenceEngine()
    
    # List available models
    print("\n📋 Available Models:")
    for model_id in engine.list_available_models():
        model_info = engine.registry.get_model(model_id)
        print(f"  • {model_id}: {model_info.name} ({model_info.parameters})")
    
    print("\n✅ PHASES 8-9 COMPLETE: S3 Streaming Loader & Inference Engine Created")
    print("\nNote: Actual inference requires downloading models from S3 (use with caution on limited disk space)")
