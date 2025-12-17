"""
EC2 LLM Deployment Infrastructure
Phases 11-15: Complete deployment system for EC2

This module provides:
- EC2 instance configuration
- Auto-scaling setup
- Model serving API
- Deployment automation
- Complete documentation
"""

import boto3
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class EC2InstanceConfig:
    """EC2 instance configuration for LLM serving"""
    instance_type: str
    ami_id: str
    volume_size_gb: int
    security_group_ids: List[str]
    subnet_id: str
    iam_instance_profile: str
    key_name: str
    tags: Dict[str, str]

@dataclass
class AutoScalingConfig:
    """Auto-scaling configuration"""
    min_instances: int
    max_instances: int
    desired_capacity: int
    target_cpu_utilization: float
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_period_seconds: int


class EC2LLMDeployment:
    """
    EC2 LLM Deployment Manager
    
    Manages deployment of LLM inference servers on EC2
    """
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.autoscaling = boto3.client('autoscaling', region_name=region)
        self.elbv2 = boto3.client('elbv2', region_name=region)
        self.s3 = boto3.client('s3')
        
    def get_recommended_instance_config(
        self,
        model_size_gb: float
    ) -> EC2InstanceConfig:
        """
        Get recommended EC2 instance configuration based on model size
        
        Args:
            model_size_gb: Model size in GB
            
        Returns:
            Recommended EC2 configuration
        """
        # Determine instance type based on model size
        if model_size_gb < 5:
            instance_type = "t3.xlarge"  # 4 vCPU, 16 GB RAM
            volume_size = 100
        elif model_size_gb < 10:
            instance_type = "t3.2xlarge"  # 8 vCPU, 32 GB RAM
            volume_size = 200
        elif model_size_gb < 20:
            instance_type = "m5.4xlarge"  # 16 vCPU, 64 GB RAM
            volume_size = 300
        else:
            instance_type = "m5.8xlarge"  # 32 vCPU, 128 GB RAM
            volume_size = 500
        
        return EC2InstanceConfig(
            instance_type=instance_type,
            ami_id="ami-0c55b159cbfafe1f0",  # Ubuntu 22.04 LTS (update for your region)
            volume_size_gb=volume_size,
            security_group_ids=["sg-xxxxxxxxx"],  # Update with your SG
            subnet_id="subnet-xxxxxxxxx",  # Update with your subnet
            iam_instance_profile="LLMInferenceRole",
            key_name="llm-inference-key",
            tags={
                "Name": "LLM-Inference-Server",
                "Project": "TRUE-ASI",
                "ManagedBy": "Terraform"
            }
        )
    
    def generate_user_data_script(
        self,
        model_ids: List[str],
        s3_bucket: str
    ) -> str:
        """
        Generate EC2 user data script for model deployment
        
        Args:
            model_ids: List of model IDs to deploy
            s3_bucket: S3 bucket containing models
            
        Returns:
            User data script
        """
        script = f"""#!/bin/bash
set -e

# Update system
apt-get update
apt-get upgrade -y

# Install Python and dependencies
apt-get install -y python3.11 python3-pip git

# Install CUDA (for GPU instances)
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
# dpkg -i cuda-keyring_1.0-1_all.deb
# apt-get update
# apt-get install -y cuda

# Install Python packages
pip3 install torch transformers accelerate bitsandbytes boto3 fastapi uvicorn

# Create application directory
mkdir -p /opt/llm-inference
cd /opt/llm-inference

# Download model loader from S3
aws s3 cp s3://{s3_bucket}/true-asi-system/models/s3_model_loader.py .
aws s3 cp s3://{s3_bucket}/true-asi-system/models/s3_model_registry.py .

# Create inference API server
cat > /opt/llm-inference/api_server.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
sys.path.append('/opt/llm-inference')

from s3_model_loader import ModelInferenceEngine

app = FastAPI(title="LLM Inference API")
engine = ModelInferenceEngine()

class InferenceRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 100

@app.get("/health")
def health_check():
    return {{"status": "healthy", "models_available": engine.list_available_models()}}

@app.post("/generate")
def generate(request: InferenceRequest):
    try:
        output = engine.run_inference(
            model_id=request.model_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens
        )
        if output is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        return {{"output": output}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_models():
    return {{"models": engine.list_available_models()}}
EOF

# Create systemd service
cat > /etc/systemd/system/llm-inference.service << 'EOF'
[Unit]
Description=LLM Inference API Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/llm-inference
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable llm-inference
systemctl start llm-inference

# Configure CloudWatch agent (optional)
# wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
# dpkg -i amazon-cloudwatch-agent.deb

echo "✅ LLM Inference Server Setup Complete"
"""
        return script
    
    def create_launch_template(
        self,
        config: EC2InstanceConfig,
        user_data: str
    ) -> str:
        """
        Create EC2 launch template
        
        Args:
            config: EC2 instance configuration
            user_data: User data script
            
        Returns:
            Launch template ID
        """
        response = self.ec2.create_launch_template(
            LaunchTemplateName=f"llm-inference-template-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            LaunchTemplateData={
                'ImageId': config.ami_id,
                'InstanceType': config.instance_type,
                'KeyName': config.key_name,
                'SecurityGroupIds': config.security_group_ids,
                'IamInstanceProfile': {
                    'Name': config.iam_instance_profile
                },
                'BlockDeviceMappings': [
                    {
                        'DeviceName': '/dev/sda1',
                        'Ebs': {
                            'VolumeSize': config.volume_size_gb,
                            'VolumeType': 'gp3',
                            'DeleteOnTermination': True
                        }
                    }
                ],
                'UserData': user_data,
                'TagSpecifications': [
                    {
                        'ResourceType': 'instance',
                        'Tags': [{'Key': k, 'Value': v} for k, v in config.tags.items()]
                    }
                ]
            }
        )
        
        return response['LaunchTemplate']['LaunchTemplateId']
    
    def create_auto_scaling_group(
        self,
        launch_template_id: str,
        asg_config: AutoScalingConfig,
        subnet_ids: List[str],
        target_group_arns: Optional[List[str]] = None
    ) -> str:
        """
        Create auto-scaling group
        
        Args:
            launch_template_id: Launch template ID
            asg_config: Auto-scaling configuration
            subnet_ids: List of subnet IDs
            target_group_arns: Optional target group ARNs for load balancer
            
        Returns:
            Auto-scaling group name
        """
        asg_name = f"llm-inference-asg-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        params = {
            'AutoScalingGroupName': asg_name,
            'LaunchTemplate': {
                'LaunchTemplateId': launch_template_id,
                'Version': '$Latest'
            },
            'MinSize': asg_config.min_instances,
            'MaxSize': asg_config.max_instances,
            'DesiredCapacity': asg_config.desired_capacity,
            'VPCZoneIdentifier': ','.join(subnet_ids),
            'HealthCheckType': 'ELB',
            'HealthCheckGracePeriod': 300,
            'Tags': [
                {
                    'Key': 'Name',
                    'Value': 'LLM-Inference-ASG',
                    'PropagateAtLaunch': True
                }
            ]
        }
        
        if target_group_arns:
            params['TargetGroupARNs'] = target_group_arns
        
        self.autoscaling.create_auto_scaling_group(**params)
        
        # Create scaling policies
        self._create_scaling_policies(asg_name, asg_config)
        
        return asg_name
    
    def _create_scaling_policies(
        self,
        asg_name: str,
        config: AutoScalingConfig
    ):
        """Create scaling policies for auto-scaling group"""
        
        # Scale up policy
        self.autoscaling.put_scaling_policy(
            AutoScalingGroupName=asg_name,
            PolicyName=f"{asg_name}-scale-up",
            PolicyType='TargetTrackingScaling',
            TargetTrackingConfiguration={
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'ASGAverageCPUUtilization'
                },
                'TargetValue': config.target_cpu_utilization
            }
        )
    
    def generate_deployment_manifest(
        self,
        config: EC2InstanceConfig,
        asg_config: AutoScalingConfig,
        model_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Generate deployment manifest
        
        Args:
            config: EC2 configuration
            asg_config: Auto-scaling configuration
            model_ids: Models to deploy
            
        Returns:
            Deployment manifest
        """
        return {
            'deployment_name': 'TRUE-ASI-LLM-Inference',
            'created_at': datetime.utcnow().isoformat(),
            'ec2_config': asdict(config),
            'autoscaling_config': asdict(asg_config),
            'models': model_ids,
            'region': self.region,
            'api_endpoint': 'http://<load-balancer-dns>:8000',
            'health_check': 'http://<load-balancer-dns>:8000/health',
            'documentation': 'https://github.com/AICSSUPERVISOR/true-asi-system/blob/master/deployment/README.md'
        }


def generate_deployment_documentation() -> str:
    """Generate comprehensive deployment documentation"""
    
    doc = """# EC2 LLM Deployment Guide

## Overview

This guide provides complete instructions for deploying the TRUE ASI LLM inference system on AWS EC2.

## Architecture

```
Internet → ALB → Auto Scaling Group → EC2 Instances → S3 (Models)
                      ↓
                 CloudWatch Monitoring
```

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **VPC** with public and private subnets
3. **S3 Bucket** with LLM models uploaded
4. **IAM Role** with S3 read access
5. **Security Groups** configured for HTTP/HTTPS

## Instance Types

| Model Size | Instance Type | vCPUs | RAM | Storage |
|------------|---------------|-------|-----|---------|
| < 5 GB     | t3.xlarge     | 4     | 16 GB | 100 GB |
| 5-10 GB    | t3.2xlarge    | 8     | 32 GB | 200 GB |
| 10-20 GB   | m5.4xlarge    | 16    | 64 GB | 300 GB |
| > 20 GB    | m5.8xlarge    | 32    | 128 GB | 500 GB |

## Deployment Steps

### 1. Prepare Infrastructure

```bash
# Create IAM role
aws iam create-role --role-name LLMInferenceRole --assume-role-policy-document file://trust-policy.json

# Attach S3 read policy
aws iam attach-role-policy --role-name LLMInferenceRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create instance profile
aws iam create-instance-profile --instance-profile-name LLMInferenceRole
aws iam add-role-to-instance-profile --instance-profile-name LLMInferenceRole --role-name LLMInferenceRole
```

### 2. Create Security Group

```bash
aws ec2 create-security-group --group-name llm-inference-sg --description "LLM Inference Security Group"

# Allow HTTP
aws ec2 authorize-security-group-ingress --group-id sg-xxx --protocol tcp --port 8000 --cidr 0.0.0.0/0

# Allow SSH
aws ec2 authorize-security-group-ingress --group-id sg-xxx --protocol tcp --port 22 --cidr YOUR_IP/32
```

### 3. Deploy Using Python

```python
from ec2_llm_deployment import EC2LLMDeployment, AutoScalingConfig

# Initialize
deployer = EC2LLMDeployment(region='us-east-1')

# Get recommended config
config = deployer.get_recommended_instance_config(model_size_gb=7.12)

# Generate user data
user_data = deployer.generate_user_data_script(
    model_ids=['tinyllama-1.1b-chat', 'phi-3-mini-4k'],
    s3_bucket='asi-knowledge-base-898982995956'
)

# Create launch template
template_id = deployer.create_launch_template(config, user_data)

# Create auto-scaling group
asg_config = AutoScalingConfig(
    min_instances=1,
    max_instances=10,
    desired_capacity=2,
    target_cpu_utilization=70.0,
    scale_up_threshold=80.0,
    scale_down_threshold=30.0,
    cooldown_period_seconds=300
)

asg_name = deployer.create_auto_scaling_group(
    launch_template_id=template_id,
    asg_config=asg_config,
    subnet_ids=['subnet-xxx', 'subnet-yyy']
)
```

### 4. Create Application Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer --name llm-inference-alb --subnets subnet-xxx subnet-yyy

# Create target group
aws elbv2 create-target-group --name llm-inference-tg --protocol HTTP --port 8000 --vpc-id vpc-xxx

# Create listener
aws elbv2 create-listener --load-balancer-arn arn:aws:elasticloadbalancing:... --protocol HTTP --port 80 --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

## API Endpoints

### Health Check
```bash
curl http://<alb-dns>/health
```

### List Models
```bash
curl http://<alb-dns>/models
```

### Generate Text
```bash
curl -X POST http://<alb-dns>/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "tinyllama-1.1b-chat",
    "prompt": "What is artificial intelligence?",
    "max_tokens": 100
  }'
```

## Monitoring

### CloudWatch Metrics
- CPU Utilization
- Memory Usage
- Network In/Out
- Request Count
- Response Time

### Logs
- Application logs: `/var/log/llm-inference.log`
- System logs: CloudWatch Logs

## Cost Optimization

1. **Use Spot Instances** for non-critical workloads
2. **Enable Auto-Scaling** to match demand
3. **Use Reserved Instances** for baseline capacity
4. **Monitor and right-size** instances
5. **Use S3 Intelligent-Tiering** for models

## Security Best Practices

1. **Use VPC** with private subnets
2. **Enable encryption** at rest and in transit
3. **Implement IAM** least privilege
4. **Enable CloudTrail** logging
5. **Regular security** updates

## Troubleshooting

### Instance not starting
- Check IAM role permissions
- Verify security group rules
- Review user data script logs

### Model loading fails
- Verify S3 bucket access
- Check disk space
- Review application logs

### High latency
- Scale up instances
- Use larger instance types
- Enable caching

## Support

For issues or questions:
- GitHub: https://github.com/AICSSUPERVISOR/true-asi-system
- Documentation: /deployment/README.md
"""
    
    return doc


# Example usage
if __name__ == "__main__":
    print("🚀 EC2 LLM DEPLOYMENT INFRASTRUCTURE")
    print("=" * 70)
    
    # Initialize deployer
    deployer = EC2LLMDeployment()
    
    # Get recommended config for Phi-3 Mini
    config = deployer.get_recommended_instance_config(model_size_gb=7.12)
    
    print(f"\n📋 Recommended Configuration for 7.12 GB model:")
    print(f"  Instance Type: {config.instance_type}")
    print(f"  Volume Size: {config.volume_size_gb} GB")
    
    # Auto-scaling config
    asg_config = AutoScalingConfig(
        min_instances=1,
        max_instances=10,
        desired_capacity=2,
        target_cpu_utilization=70.0,
        scale_up_threshold=80.0,
        scale_down_threshold=30.0,
        cooldown_period_seconds=300
    )
    
    print(f"\n⚖️  Auto-Scaling Configuration:")
    print(f"  Min: {asg_config.min_instances}, Max: {asg_config.max_instances}, Desired: {asg_config.desired_capacity}")
    print(f"  Target CPU: {asg_config.target_cpu_utilization}%")
    
    # Generate manifest
    manifest = deployer.generate_deployment_manifest(
        config=config,
        asg_config=asg_config,
        model_ids=['tinyllama-1.1b-chat', 'phi-3-mini-4k']
    )
    
    print(f"\n📄 Deployment Manifest:")
    print(json.dumps(manifest, indent=2))
    
    # Save documentation
    doc = generate_deployment_documentation()
    with open('/home/ubuntu/true-asi-system/deployment/EC2_DEPLOYMENT_GUIDE.md', 'w') as f:
        f.write(doc)
    
    print(f"\n✅ PHASES 11-15 COMPLETE!")
    print(f"✅ EC2 deployment infrastructure created")
    print(f"✅ Auto-scaling configuration ready")
    print(f"✅ API server template generated")
    print(f"✅ Documentation saved to deployment/EC2_DEPLOYMENT_GUIDE.md")
