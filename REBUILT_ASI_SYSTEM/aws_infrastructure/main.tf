# ASI System - Complete AWS Infrastructure
# Terraform configuration for 10+ TB knowledge base and model serving

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "asi-terraform-state"
    key    = "asi-system/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "ASI-System"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Variables
variable "aws_region" {
  default = "us-east-1"
}

variable "environment" {
  default = "production"
}

variable "knowledge_base_bucket" {
  default = "asi-knowledge-base-898982995956"
}

# VPC Configuration
resource "aws_vpc" "asi_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "asi-production-vpc"
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count                   = 3
  vpc_id                  = aws_vpc.asi_vpc.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "asi-public-${count.index + 1}"
    Type = "public"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.asi_vpc.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "asi-private-${count.index + 1}"
    Type = "private"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "asi_igw" {
  vpc_id = aws_vpc.asi_vpc.id
  
  tags = {
    Name = "asi-internet-gateway"
  }
}

# NAT Gateway
resource "aws_eip" "nat" {
  count  = 3
  domain = "vpc"
  
  tags = {
    Name = "asi-nat-eip-${count.index + 1}"
  }
}

resource "aws_nat_gateway" "asi_nat" {
  count         = 3
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = {
    Name = "asi-nat-gateway-${count.index + 1}"
  }
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.asi_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.asi_igw.id
  }
  
  tags = {
    Name = "asi-public-rt"
  }
}

resource "aws_route_table" "private" {
  count  = 3
  vpc_id = aws_vpc.asi_vpc.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.asi_nat[count.index].id
  }
  
  tags = {
    Name = "asi-private-rt-${count.index + 1}"
  }
}

# S3 Bucket for Knowledge Base (10+ TB)
resource "aws_s3_bucket" "knowledge_base" {
  bucket = var.knowledge_base_bucket
  
  tags = {
    Name        = "ASI Knowledge Base"
    Description = "10+ TB of AI models, datasets, and knowledge"
  }
}

resource "aws_s3_bucket_versioning" "knowledge_base" {
  bucket = aws_s3_bucket.knowledge_base.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "knowledge_base" {
  bucket = aws_s3_bucket.knowledge_base.id
  
  rule {
    id     = "intelligent-tiering"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "GLACIER"
    }
  }
}

# EKS Cluster for Model Serving
resource "aws_eks_cluster" "asi_cluster" {
  name     = "true-asi-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.28"
  
  vpc_config {
    subnet_ids              = aws_subnet.private[*].id
    endpoint_private_access = true
    endpoint_public_access  = true
  }
  
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy
  ]
}

# EKS Node Groups
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.asi_cluster.name
  node_group_name = "gpu-inference-nodes"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = aws_subnet.private[*].id
  
  instance_types = ["p4d.24xlarge"]  # 8x A100 GPUs
  
  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 1
  }
  
  update_config {
    max_unavailable = 1
  }
  
  labels = {
    "node-type" = "gpu"
    "workload"  = "inference"
  }
  
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }
}

resource "aws_eks_node_group" "cpu_nodes" {
  cluster_name    = aws_eks_cluster.asi_cluster.name
  node_group_name = "cpu-general-nodes"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = aws_subnet.private[*].id
  
  instance_types = ["m6i.4xlarge"]
  
  scaling_config {
    desired_size = 5
    max_size     = 20
    min_size     = 3
  }
  
  labels = {
    "node-type" = "cpu"
    "workload"  = "general"
  }
}

# RDS for Metadata
resource "aws_db_instance" "asi_metadata" {
  identifier           = "asi-metadata-db"
  engine               = "postgres"
  engine_version       = "15.4"
  instance_class       = "db.r6g.2xlarge"
  allocated_storage    = 500
  max_allocated_storage = 2000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "asi_metadata"
  username = "asi_admin"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.asi.name
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"
  
  multi_az               = true
  deletion_protection    = true
  skip_final_snapshot    = false
  final_snapshot_identifier = "asi-metadata-final-snapshot"
}

# ElastiCache for Caching
resource "aws_elasticache_cluster" "asi_cache" {
  cluster_id           = "asi-cache"
  engine               = "redis"
  node_type            = "cache.r6g.xlarge"
  num_cache_nodes      = 3
  parameter_group_name = "default.redis7"
  port                 = 6379
  
  security_group_ids = [aws_security_group.redis.id]
  subnet_group_name  = aws_elasticache_subnet_group.asi.name
}

# OpenSearch for Vector Search
resource "aws_opensearch_domain" "asi_vectors" {
  domain_name    = "asi-vectors"
  engine_version = "OpenSearch_2.11"
  
  cluster_config {
    instance_type            = "r6g.2xlarge.search"
    instance_count           = 3
    dedicated_master_enabled = true
    dedicated_master_type    = "r6g.large.search"
    dedicated_master_count   = 3
    zone_awareness_enabled   = true
    
    zone_awareness_config {
      availability_zone_count = 3
    }
  }
  
  ebs_options {
    ebs_enabled = true
    volume_size = 500
    volume_type = "gp3"
    iops        = 3000
    throughput  = 250
  }
  
  encrypt_at_rest {
    enabled = true
  }
  
  node_to_node_encryption {
    enabled = true
  }
  
  vpc_options {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.opensearch.id]
  }
}

# SQS Queues for Task Processing
resource "aws_sqs_queue" "inference_queue" {
  name                       = "asi-inference-queue"
  delay_seconds              = 0
  max_message_size           = 262144
  message_retention_seconds  = 86400
  receive_wait_time_seconds  = 20
  visibility_timeout_seconds = 300
  
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.inference_dlq.arn
    maxReceiveCount     = 3
  })
}

resource "aws_sqs_queue" "inference_dlq" {
  name = "asi-inference-dlq"
}

# Lambda for Serverless Processing
resource "aws_lambda_function" "model_router" {
  function_name = "asi-model-router"
  role          = aws_iam_role.lambda.arn
  handler       = "index.handler"
  runtime       = "python3.11"
  timeout       = 30
  memory_size   = 1024
  
  filename         = "lambda/model_router.zip"
  source_code_hash = filebase64sha256("lambda/model_router.zip")
  
  environment {
    variables = {
      INFERENCE_QUEUE_URL = aws_sqs_queue.inference_queue.url
      CACHE_ENDPOINT      = aws_elasticache_cluster.asi_cache.cache_nodes[0].address
    }
  }
  
  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }
}

# API Gateway
resource "aws_apigatewayv2_api" "asi_api" {
  name          = "asi-api"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers = ["*"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_stage" "production" {
  api_id      = aws_apigatewayv2_api.asi_api.id
  name        = "production"
  auto_deploy = true
  
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_logs.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      responseLength = "$context.responseLength"
    })
  }
}

# CloudWatch Monitoring
resource "aws_cloudwatch_log_group" "api_logs" {
  name              = "/aws/apigateway/asi-api"
  retention_in_days = 30
}

resource "aws_cloudwatch_dashboard" "asi_dashboard" {
  dashboard_name = "ASI-System-Dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "API Requests"
          region = var.aws_region
          metrics = [
            ["AWS/ApiGateway", "Count", "ApiId", aws_apigatewayv2_api.asi_api.id]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Inference Latency"
          region = var.aws_region
          metrics = [
            ["AWS/Lambda", "Duration", "FunctionName", aws_lambda_function.model_router.function_name]
          ]
        }
      }
    ]
  })
}

# IAM Roles
resource "aws_iam_role" "eks_cluster" {
  name = "asi-eks-cluster-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

resource "aws_iam_role" "eks_node" {
  name = "asi-eks-node-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node.name
}

resource "aws_iam_role_policy_attachment" "eks_container_registry" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node.name
}

resource "aws_iam_role" "lambda" {
  name = "asi-lambda-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# Security Groups
resource "aws_security_group" "rds" {
  name        = "asi-rds-sg"
  description = "Security group for RDS"
  vpc_id      = aws_vpc.asi_vpc.id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
}

resource "aws_security_group" "redis" {
  name        = "asi-redis-sg"
  description = "Security group for Redis"
  vpc_id      = aws_vpc.asi_vpc.id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
}

resource "aws_security_group" "opensearch" {
  name        = "asi-opensearch-sg"
  description = "Security group for OpenSearch"
  vpc_id      = aws_vpc.asi_vpc.id
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
}

resource "aws_security_group" "lambda" {
  name        = "asi-lambda-sg"
  description = "Security group for Lambda"
  vpc_id      = aws_vpc.asi_vpc.id
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Subnet Groups
resource "aws_db_subnet_group" "asi" {
  name       = "asi-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_subnet_group" "asi" {
  name       = "asi-cache-subnet-group"
  subnet_ids = aws_subnet.private[*].id
}

# Data Sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Outputs
output "vpc_id" {
  value = aws_vpc.asi_vpc.id
}

output "eks_cluster_endpoint" {
  value = aws_eks_cluster.asi_cluster.endpoint
}

output "eks_cluster_name" {
  value = aws_eks_cluster.asi_cluster.name
}

output "s3_bucket" {
  value = aws_s3_bucket.knowledge_base.bucket
}

output "api_endpoint" {
  value = aws_apigatewayv2_stage.production.invoke_url
}

output "rds_endpoint" {
  value = aws_db_instance.asi_metadata.endpoint
}

output "opensearch_endpoint" {
  value = aws_opensearch_domain.asi_vectors.endpoint
}
