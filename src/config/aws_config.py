"""AWS Configuration"""
import os

AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
S3_BUCKET = os.getenv('S3_BUCKET', 'asi-knowledge-base-898982995956')

# DynamoDB Configuration
DYNAMODB_CONFIG = {
    'region_name': AWS_REGION,
    'entities_table': os.getenv('DYNAMODB_ENTITIES_TABLE', 'asi-knowledge-graph-entities'),
    'relationships_table': os.getenv('DYNAMODB_RELATIONSHIPS_TABLE', 'asi-knowledge-graph-relationships'),
    'agents_table': os.getenv('DYNAMODB_AGENTS_TABLE', 'multi-agent-asi-system')
}

# S3 Configuration
S3_CONFIG = {
    'bucket': S3_BUCKET,
    'region': AWS_REGION,
    'prefixes': {
        'repositories': 'repositories/',
        'entities': 'entities/',
        'results': 'maximum_power_results/',
        'code': 'proprietary_code_full/'
    }
}

# SQS Configuration
SQS_CONFIG = {
    'queue_url': os.getenv('SQS_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/898982995956/asi-agent-tasks'),
    'region': AWS_REGION,
    'max_messages': 10,
    'wait_time': 20
}
