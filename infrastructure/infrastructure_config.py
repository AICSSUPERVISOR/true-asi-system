"""
Infrastructure Configuration for S-7 ASI
Terraform, Kubernetes, Docker, and cloud deployment automation
Part of the TRUE ASI System - 100/100 Quality
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    RUNPOD = "runpod"


class InfrastructureComponent(Enum):
    """Infrastructure components"""
    COMPUTE = "compute"
    STORAGE = "storage"
    DATABASE = "database"
    NETWORKING = "networking"
    MONITORING = "monitoring"


@dataclass
class TerraformConfig:
    """Terraform configuration"""
    provider: CloudProvider
    region: str
    instance_type: str
    instance_count: int
    disk_size_gb: int
    vpc_cidr: str = "10.0.0.0/16"
    enable_monitoring: bool = True
    enable_backup: bool = True
    tags: Dict[str, str] = None


@dataclass
class KubernetesConfig:
    """Kubernetes configuration"""
    cluster_name: str
    node_count: int
    node_instance_type: str
    kubernetes_version: str = "1.28"
    enable_autoscaling: bool = True
    min_nodes: int = 1
    max_nodes: int = 10
    enable_gpu: bool = True


@dataclass
class DockerConfig:
    """Docker configuration"""
    base_image: str
    python_version: str = "3.11"
    cuda_version: Optional[str] = "12.1"
    install_packages: List[str] = None
    environment_vars: Dict[str, str] = None


class TerraformGenerator:
    """Generate Terraform configurations"""
    
    def __init__(self, config: TerraformConfig):
        self.config = config
        
    def generate_provider_config(self) -> str:
        """Generate provider configuration"""
        if self.config.provider == CloudProvider.AWS:
            return f'''
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = "{self.config.region}"
}}
'''
        elif self.config.provider == CloudProvider.GCP:
            return f'''
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 5.0"
    }}
  }}
}}

provider "google" {{
  region = "{self.config.region}"
}}
'''
        return ""
    
    def generate_compute_config(self) -> str:
        """Generate compute resources"""
        if self.config.provider == CloudProvider.AWS:
            tags_str = json.dumps(self.config.tags or {"Project": "S7-ASI"})
            return f'''
resource "aws_instance" "asi_compute" {{
  count         = {self.config.instance_count}
  ami           = data.aws_ami.ubuntu.id
  instance_type = "{self.config.instance_type}"
  
  root_block_device {{
    volume_size = {self.config.disk_size_gb}
    volume_type = "gp3"
  }}
  
  tags = {tags_str}
}}

data "aws_ami" "ubuntu" {{
  most_recent = true
  owners      = ["099720109477"]  # Canonical
  
  filter {{
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }}
}}
'''
        return ""
    
    def generate_storage_config(self) -> str:
        """Generate storage resources"""
        if self.config.provider == CloudProvider.AWS:
            return f'''
resource "aws_s3_bucket" "asi_storage" {{
  bucket = "asi-knowledge-base-${{random_id.bucket_suffix.hex}}"
  
  tags = {{
    Name = "S7-ASI-Storage"
  }}
}}

resource "aws_s3_bucket_versioning" "asi_storage" {{
  bucket = aws_s3_bucket.asi_storage.id
  
  versioning_configuration {{
    status = "Enabled"
  }}
}}

resource "random_id" "bucket_suffix" {{
  byte_length = 8
}}
'''
        return ""
    
    def generate_networking_config(self) -> str:
        """Generate networking resources"""
        if self.config.provider == CloudProvider.AWS:
            return f'''
resource "aws_vpc" "asi_vpc" {{
  cidr_block           = "{self.config.vpc_cidr}"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {{
    Name = "S7-ASI-VPC"
  }}
}}

resource "aws_subnet" "asi_subnet" {{
  vpc_id            = aws_vpc.asi_vpc.id
  cidr_block        = cidrsubnet(aws_vpc.asi_vpc.cidr_block, 8, 1)
  availability_zone = data.aws_availability_zones.available.names[0]
  
  tags = {{
    Name = "S7-ASI-Subnet"
  }}
}}

data "aws_availability_zones" "available" {{
  state = "available"
}}

resource "aws_internet_gateway" "asi_igw" {{
  vpc_id = aws_vpc.asi_vpc.id
  
  tags = {{
    Name = "S7-ASI-IGW"
  }}
}}
'''
        return ""
    
    def generate_full_config(self) -> str:
        """Generate complete Terraform configuration"""
        config = []
        config.append(self.generate_provider_config())
        config.append(self.generate_compute_config())
        config.append(self.generate_storage_config())
        config.append(self.generate_networking_config())
        
        return "\n".join(config)
    
    def save_config(self, output_dir: str = "./terraform"):
        """Save Terraform configuration to files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save main configuration
        with open(f"{output_dir}/main.tf", "w") as f:
            f.write(self.generate_full_config())
        
        return output_dir


class KubernetesGenerator:
    """Generate Kubernetes configurations"""
    
    def __init__(self, config: KubernetesConfig):
        self.config = config
        
    def generate_deployment(self, 
                           name: str,
                           image: str,
                           replicas: int = 1,
                           resources: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate Kubernetes deployment"""
        resources = resources or {
            "requests": {"cpu": "1", "memory": "2Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"}
        }
        
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": name,
                "labels": {"app": name}
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {"app": name}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": name}
                    },
                    "spec": {
                        "containers": [{
                            "name": name,
                            "image": image,
                            "resources": resources,
                            "ports": [{"containerPort": 8000}]
                        }]
                    }
                }
            }
        }
        
        return deployment
    
    def generate_service(self, 
                        name: str,
                        port: int = 80,
                        target_port: int = 8000,
                        service_type: str = "LoadBalancer") -> Dict[str, Any]:
        """Generate Kubernetes service"""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{name}-service"
            },
            "spec": {
                "type": service_type,
                "selector": {"app": name},
                "ports": [{
                    "protocol": "TCP",
                    "port": port,
                    "targetPort": target_port
                }]
            }
        }
        
        return service
    
    def generate_hpa(self,
                    name: str,
                    min_replicas: int = 1,
                    max_replicas: int = 10,
                    cpu_threshold: int = 70) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler"""
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{name}-hpa"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": name
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": cpu_threshold
                        }
                    }
                }]
            }
        }
        
        return hpa
    
    def save_configs(self, output_dir: str = "./kubernetes"):
        """Save Kubernetes configurations"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate configurations for S-7 components
        components = [
            {"name": "asi-api", "image": "asi-api:latest", "replicas": 3},
            {"name": "asi-worker", "image": "asi-worker:latest", "replicas": 10},
            {"name": "asi-scheduler", "image": "asi-scheduler:latest", "replicas": 1}
        ]
        
        for component in components:
            # Deployment
            deployment = self.generate_deployment(
                name=component["name"],
                image=component["image"],
                replicas=component["replicas"]
            )
            
            with open(f"{output_dir}/{component['name']}-deployment.yaml", "w") as f:
                yaml.dump(deployment, f, default_flow_style=False)
            
            # Service
            service = self.generate_service(name=component["name"])
            
            with open(f"{output_dir}/{component['name']}-service.yaml", "w") as f:
                yaml.dump(service, f, default_flow_style=False)
            
            # HPA
            if component["name"] != "asi-scheduler":
                hpa = self.generate_hpa(
                    name=component["name"],
                    min_replicas=self.config.min_nodes,
                    max_replicas=self.config.max_nodes
                )
                
                with open(f"{output_dir}/{component['name']}-hpa.yaml", "w") as f:
                    yaml.dump(hpa, f, default_flow_style=False)
        
        return output_dir


class DockerfileGenerator:
    """Generate Dockerfiles"""
    
    def __init__(self, config: DockerConfig):
        self.config = config
        
    def generate_dockerfile(self) -> str:
        """Generate Dockerfile content"""
        packages = self.config.install_packages or [
            "torch",
            "transformers",
            "accelerate",
            "deepspeed",
            "ray",
            "fastapi",
            "uvicorn"
        ]
        
        env_vars = self.config.environment_vars or {}
        
        dockerfile = f'''# S-7 ASI Docker Image
FROM {self.config.base_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
'''
        
        # Add custom environment variables
        for key, value in env_vars.items():
            dockerfile += f'ENV {key}={value}\n'
        
        dockerfile += f'''
# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    vim \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python {self.config.python_version}
RUN apt-get update && apt-get install -y python{self.config.python_version} python3-pip

# Install CUDA (if specified)
'''
        
        if self.config.cuda_version:
            dockerfile += f'''RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \\
    && mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 \\
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \\
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \\
    && apt-get update \\
    && apt-get install -y cuda-{self.config.cuda_version.replace(".", "-")}
'''
        
        dockerfile += f'''
# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional packages
RUN pip3 install --no-cache-dir {" ".join(packages)}

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python3", "main.py"]
'''
        
        return dockerfile
    
    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml"""
        compose = '''version: '3.8'

services:
  asi-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SERVICE_TYPE=api
    volumes:
      - ./data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  
  asi-worker:
    build: .
    environment:
      - SERVICE_TYPE=worker
    volumes:
      - ./data:/app/data
    deploy:
      replicas: 4
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  
  asi-scheduler:
    build: .
    environment:
      - SERVICE_TYPE=scheduler
    volumes:
      - ./data:/app/data
'''
        
        return compose
    
    def save_files(self, output_dir: str = "./docker"):
        """Save Docker configuration files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save Dockerfile
        with open(f"{output_dir}/Dockerfile", "w") as f:
            f.write(self.generate_dockerfile())
        
        # Save docker-compose.yml
        with open(f"{output_dir}/docker-compose.yml", "w") as f:
            f.write(self.generate_docker_compose())
        
        # Save requirements.txt
        requirements = [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "deepspeed>=0.10.0",
            "ray[default]>=2.5.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "pydantic>=2.0.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0"
        ]
        
        with open(f"{output_dir}/requirements.txt", "w") as f:
            f.write("\n".join(requirements))
        
        return output_dir


class InfrastructureManager:
    """Unified infrastructure management"""
    
    def __init__(self, 
                 terraform_config: TerraformConfig,
                 kubernetes_config: KubernetesConfig,
                 docker_config: DockerConfig):
        self.terraform_config = terraform_config
        self.kubernetes_config = kubernetes_config
        self.docker_config = docker_config
        
        # Initialize generators
        self.terraform_gen = TerraformGenerator(terraform_config)
        self.kubernetes_gen = KubernetesGenerator(kubernetes_config)
        self.docker_gen = DockerfileGenerator(docker_config)
        
    def generate_all_configs(self, output_dir: str = "./infrastructure"):
        """Generate all infrastructure configurations"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate Terraform configs
        terraform_dir = self.terraform_gen.save_config(f"{output_dir}/terraform")
        
        # Generate Kubernetes configs
        k8s_dir = self.kubernetes_gen.save_configs(f"{output_dir}/kubernetes")
        
        # Generate Docker configs
        docker_dir = self.docker_gen.save_files(f"{output_dir}/docker")
        
        # Generate deployment script
        deploy_script = f'''#!/bin/bash
# S-7 ASI Infrastructure Deployment Script

echo "Deploying S-7 ASI Infrastructure..."

# Deploy Terraform
echo "1. Deploying cloud infrastructure..."
cd {terraform_dir}
terraform init
terraform plan
terraform apply -auto-approve

# Build Docker images
echo "2. Building Docker images..."
cd ../{docker_dir}
docker-compose build

# Deploy Kubernetes
echo "3. Deploying Kubernetes resources..."
cd ../{k8s_dir}
kubectl apply -f .

echo "Deployment complete!"
'''
        
        with open(f"{output_dir}/deploy.sh", "w") as f:
            f.write(deploy_script)
        
        os.chmod(f"{output_dir}/deploy.sh", 0o755)
        
        return {
            'terraform_dir': terraform_dir,
            'kubernetes_dir': k8s_dir,
            'docker_dir': docker_dir,
            'deploy_script': f"{output_dir}/deploy.sh"
        }


# Example usage
if __name__ == "__main__":
    # Configure infrastructure
    terraform_config = TerraformConfig(
        provider=CloudProvider.AWS,
        region="us-east-1",
        instance_type="p3.8xlarge",
        instance_count=10,
        disk_size_gb=500,
        tags={"Project": "S7-ASI", "Environment": "Production"}
    )
    
    kubernetes_config = KubernetesConfig(
        cluster_name="s7-asi-cluster",
        node_count=10,
        node_instance_type="p3.8xlarge",
        enable_gpu=True,
        min_nodes=5,
        max_nodes=50
    )
    
    docker_config = DockerConfig(
        base_image="nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
        python_version="3.11",
        cuda_version="12.1"
    )
    
    # Generate all configurations
    manager = InfrastructureManager(
        terraform_config,
        kubernetes_config,
        docker_config
    )
    
    result = manager.generate_all_configs("./s7-infrastructure")
    print(f"Infrastructure configs generated: {json.dumps(result, indent=2)}")
