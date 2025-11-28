# EC2 LLM Deployment Guide

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
curl -X POST http://<alb-dns>/generate   -H "Content-Type: application/json"   -d '{
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
