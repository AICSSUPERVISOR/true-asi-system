# TRUE ASI System - Deployment Guide

## 1. Docker Deployment

### Prerequisites
- Docker
- Docker Compose

### Steps

1.  **Build Images**:
    ```bash
    docker-compose build
    ```
2.  **Start Services**:
    ```bash
    docker-compose up -d
    ```

## 2. Kubernetes Deployment

### Prerequisites
- Kubernetes cluster
- `kubectl` configured

### Steps

1.  **Apply Configurations**:
    ```bash
    kubectl apply -f deployment/kubernetes/
    ```

## 3. AWS Deployment (Terraform)

### Prerequisites
- Terraform
- AWS credentials configured

### Steps

1.  **Initialize Terraform**:
    ```bash
    cd deployment/terraform
    terraform init
    ```
2.  **Apply Infrastructure**:
    ```bash
    terraform apply
    ```

