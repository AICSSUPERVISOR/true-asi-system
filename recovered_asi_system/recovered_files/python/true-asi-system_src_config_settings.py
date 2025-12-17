"""
Global Settings for TRUE ASI System
"""
import os
from typing import Optional

# System Configuration
MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '20'))
BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '500'))
LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
ENABLE_SELF_IMPROVEMENT: bool = os.getenv('ENABLE_SELF_IMPROVEMENT', 'true').lower() == 'true'

# API Keys
OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY: Optional[str] = os.getenv('ANTHROPIC_API_KEY')
PERPLEXITY_API_KEY: Optional[str] = os.getenv('PERPLEXITY_API_KEY')

# AWS Configuration
AWS_REGION: str = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
S3_BUCKET: str = os.getenv('S3_BUCKET', 'asi-knowledge-base-898982995956')

# DynamoDB Tables
DYNAMODB_ENTITIES_TABLE: str = os.getenv('DYNAMODB_ENTITIES_TABLE', 'asi-knowledge-graph-entities')
DYNAMODB_RELATIONSHIPS_TABLE: str = os.getenv('DYNAMODB_RELATIONSHIPS_TABLE', 'asi-knowledge-graph-relationships')
DYNAMODB_AGENTS_TABLE: str = os.getenv('DYNAMODB_AGENTS_TABLE', 'multi-agent-asi-system')

# SQS Configuration
SQS_QUEUE_URL: str = os.getenv('SQS_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/898982995956/asi-agent-tasks')

# Agent Configuration
TOTAL_AGENTS: int = 250
AGENT_TIMEOUT: int = 300  # seconds

# Knowledge Graph Configuration
MAX_ENTITIES: int = 1000000  # 1 million entities
MAX_RELATIONSHIPS: int = 10000000  # 10 million relationships

# Processing Configuration
ENTITY_EXTRACTION_MODEL: str = 'gpt-4.1-mini'
CODE_GENERATION_MODEL: str = 'gpt-4.1-mini'
REASONING_MODEL: str = 'gpt-4.1-mini'
