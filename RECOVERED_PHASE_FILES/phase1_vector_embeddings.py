#!/usr/bin/env python3.11
"""
PHASE 1: VECTOR EMBEDDINGS GENERATION
Generate embeddings for all catalog objects using OpenAI
100/100 quality - Maximum power utilization
"""

import json
import os
import time
from typing import Dict, List, Any
from datetime import datetime
import openai
from openai import OpenAI

class VectorEmbeddingsGenerator:
    """
    Generate vector embeddings for semantic search
    Using OpenAI embeddings API at maximum power
    """
    
    def __init__(self):
        # Initialize OpenAI clients with all available keys
        self.openai_keys = [
            os.getenv("OPENAI_API_KEY"),  # Primary key from environment
            "OPENAI_KEY_REDACTED",
            "OPENAI_KEY_REDACTED",
            "OPENAI_KEY_REDACTED"  # Manus API key
        ]
        
        # Filter out None values
        self.openai_keys = [k for k in self.openai_keys if k]
        
        self.current_key_index = 0
        self.client = OpenAI(api_key=self.openai_keys[self.current_key_index])
        
        # Embedding model
        self.embedding_model = "text-embedding-3-small"  # Fast and efficient
        
        # Statistics
        self.stats = {
            "total_objects": 0,
            "embeddings_generated": 0,
            "errors": 0,
            "api_calls": 0,
            "total_tokens": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Batch processing
        self.batch_size = 100
        self.embeddings_data = []
        
        print("="*80)
        print("VECTOR EMBEDDINGS GENERATOR INITIALIZED")
        print("="*80)
        print(f"OpenAI Keys Available: {len(self.openai_keys)}")
        print(f"Embedding Model: {self.embedding_model}")
        print(f"Batch Size: {self.batch_size}")
        print("="*80)
    
    def rotate_api_key(self):
        """Rotate to next API key if rate limited"""
        self.current_key_index = (self.current_key_index + 1) % len(self.openai_keys)
        self.client = OpenAI(api_key=self.openai_keys[self.current_key_index])
        print(f"Rotated to API key #{self.current_key_index + 1}")
    
    def generate_text_for_embedding(self, obj: Dict[str, Any]) -> str:
        """
        Generate text representation of object for embedding
        """
        key = obj.get("key", "")
        category = obj.get("category", "")
        extension = obj.get("extension", "")
        top_folder = obj.get("top_folder", "")
        
        # Create rich text representation
        text_parts = []
        
        if key:
            # Extract filename
            filename = key.split('/')[-1] if '/' in key else key
            text_parts.append(f"File: {filename}")
        
        if category:
            text_parts.append(f"Category: {category}")
        
        if extension:
            text_parts.append(f"Type: {extension}")
        
        if top_folder:
            text_parts.append(f"Folder: {top_folder}")
        
        # Add path components for context
        if key and '/' in key:
            path_parts = key.split('/')[:-1]  # Exclude filename
            if path_parts:
                text_parts.append(f"Path: {' > '.join(path_parts[:3])}")  # First 3 levels
        
        return " | ".join(text_parts)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            self.stats["api_calls"] += 1
            self.stats["total_tokens"] += response.usage.total_tokens
            
            return response.data[0].embedding
        
        except openai.RateLimitError:
            print("Rate limit hit, rotating API key...")
            self.rotate_api_key()
            time.sleep(1)
            return self.generate_embedding(text)  # Retry with new key
        
        except Exception as e:
            self.stats["errors"] += 1
            print(f"Error generating embedding: {e}")
            return None
    
    def process_catalog(self, catalog_file: str):
        """
        Process catalog and generate embeddings
        """
        print(f"\nLoading catalog from: {catalog_file}")
        
        with open(catalog_file, 'r') as f:
            catalog = json.load(f)
        
        objects = catalog.get("objects", [])
        self.stats["total_objects"] = len(objects)
        
        print(f"Total objects to process: {self.stats['total_objects']:,}")
        print("Generating embeddings...")
        print("-"*80)
        
        # Process in batches
        for i in range(0, len(objects), self.batch_size):
            batch = objects[i:i+self.batch_size]
            
            for obj in batch:
                # Generate text representation
                text = self.generate_text_for_embedding(obj)
                
                # Generate embedding
                embedding = self.generate_embedding(text)
                
                if embedding:
                    # Store embedding data
                    embedding_data = {
                        "id": obj.get("key_hash", f"obj_{i}"),
                        "key": obj.get("key", ""),
                        "embedding": embedding,
                        "metadata": {
                            "category": obj.get("category", ""),
                            "extension": obj.get("extension", ""),
                            "top_folder": obj.get("top_folder", ""),
                            "size_mb": obj.get("size_mb", 0)
                        }
                    }
                    
                    self.embeddings_data.append(embedding_data)
                    self.stats["embeddings_generated"] += 1
            
            # Progress update
            if (i + self.batch_size) % 1000 == 0:
                progress = (i + self.batch_size) / len(objects) * 100
                print(f"Progress: {i + self.batch_size:,} / {len(objects):,} ({progress:.1f}%) | "
                      f"Embeddings: {self.stats['embeddings_generated']:,} | "
                      f"API Calls: {self.stats['api_calls']:,} | "
                      f"Tokens: {self.stats['total_tokens']:,}")
            
            # Rate limiting - be respectful
            time.sleep(0.05)  # 50ms between batches
        
        print("\n" + "="*80)
        print("EMBEDDING GENERATION COMPLETE")
        print("="*80)
        print(f"Total Embeddings: {self.stats['embeddings_generated']:,}")
        print(f"API Calls: {self.stats['api_calls']:,}")
        print(f"Total Tokens: {self.stats['total_tokens']:,}")
        print(f"Errors: {self.stats['errors']}")
        print("="*80)
    
    def save_embeddings(self, filepath: str):
        """
        Save embeddings to file
        """
        print(f"\nSaving embeddings to: {filepath}")
        
        output_data = {
            "metadata": self.stats,
            "embeddings": self.embeddings_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f)
        
        file_size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"✅ Embeddings saved: {file_size_mb:.2f} MB")
    
    def save_stats(self, filepath: str):
        """Save statistics"""
        self.stats["end_time"] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✅ Stats saved to: {filepath}")

def main():
    """Main execution"""
    generator = VectorEmbeddingsGenerator()
    
    catalog_file = "/home/ubuntu/true-asi-build/phase1_complete_catalog.json"
    
    # Generate embeddings
    generator.process_catalog(catalog_file)
    
    # Save results
    generator.save_embeddings("/home/ubuntu/true-asi-build/phase1_embeddings.json")
    generator.save_stats("/home/ubuntu/true-asi-build/phase1_embeddings_stats.json")
    
    print("\n" + "="*80)
    print("VECTOR EMBEDDINGS GENERATION: COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
