"""AWS Integration Module"""
import boto3
import logging
from typing import Dict, List, Any
from ..config.aws_config import DYNAMODB_CONFIG, S3_CONFIG, SQS_CONFIG

logger = logging.getLogger(__name__)


class AWSIntegration:
    """Handles all AWS service integrations"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=S3_CONFIG['region'])
        self.dynamodb = boto3.resource('dynamodb', region_name=DYNAMODB_CONFIG['region_name'])
        self.sqs = boto3.client('sqs', region_name=SQS_CONFIG['region'])
        
        # DynamoDB tables
        self.entities_table = self.dynamodb.Table(DYNAMODB_CONFIG['entities_table'])
        self.relationships_table = self.dynamodb.Table(DYNAMODB_CONFIG['relationships_table'])
        self.agents_table = self.dynamodb.Table(DYNAMODB_CONFIG['agents_table'])
        
        logger.info("AWS Integration initialized")
    
    async def load_entities(self) -> List[Dict]:
        """Load entities from DynamoDB"""
        try:
            response = self.entities_table.scan(Limit=1000)
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Failed to load entities: {str(e)}")
            return []
    
    async def store_entity(self, entity: Dict[str, Any]):
        """Store entity in DynamoDB"""
        try:
            self.entities_table.put_item(Item=entity)
        except Exception as e:
            logger.error(f"Failed to store entity: {str(e)}")
    
    async def upload_to_s3(self, data: bytes, key: str):
        """Upload data to S3"""
        try:
            self.s3_client.put_object(
                Bucket=S3_CONFIG['bucket'],
                Key=key,
                Body=data
            )
        except Exception as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
