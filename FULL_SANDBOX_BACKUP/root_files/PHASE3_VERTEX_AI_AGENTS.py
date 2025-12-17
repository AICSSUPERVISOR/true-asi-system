#!/usr/bin/env python3.11
"""
PHASE 3: VERTEX AI INTEGRATION + AGENT DEPLOYMENT
Goal: 62/100 → 85/100
Integrate Vertex AI and deploy working agent system
"""

import os
import json
import boto3
import requests
import zipfile
from datetime import datetime
from typing import Dict, List

class Phase3VertexAIAgents:
    """Integrate Vertex AI and deploy agents"""
    
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.lambda_client = boto3.client('lambda')
        self.sqs = boto3.client('sqs')
        self.bucket = 'asi-knowledge-base-898982995956'
        
        # Vertex AI configuration
        self.vertex_api_key = 'AQ.Ab8RN6J09J-LtGcl3r7aigIc4RGi3mhE3BVk0MLdHzU2p880_g'
        self.vertex_service_account = 'vertex-express@potent-howl-464621-g7.iam.gserviceaccount.com'
        self.vertex_endpoint = 'https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:streamGenerateContent'
        
        self.results = {
            'phase': 3,
            'goal': '62 → 85',
            'timestamp': datetime.now().isoformat(),
            'vertex_ai': {},
            'agents': {},
            'score_before': 62,
            'score_after': 62
        }
    
    def test_vertex_ai(self) -> Dict:
        """Test Vertex AI API"""
        print("\n[1/6] Testing Vertex AI API...")
        
        try:
            response = requests.post(
                f"{self.vertex_endpoint}?key={self.vertex_api_key}",
                headers={'Content-Type': 'application/json'},
                json={
                    'contents': [{
                        'role': 'user',
                        'parts': [{'text': 'Reply with: VERTEX AI OPERATIONAL'}]
                    }]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                # Parse streaming response
                result = response.json()[0]
                text = result['candidates'][0]['content']['parts'][0]['text']
                model = result['modelVersion']
                
                print(f"   ✅ PASS: {text}")
                print(f"   Model: {model}")
                
                self.results['vertex_ai'] = {
                    'status': 'WORKING',
                    'model': model,
                    'response': text
                }
                return {'status': 'PASS', 'model': model}
            else:
                print(f"   ❌ FAIL: Status {response.status_code}")
                return {'status': 'FAIL', 'error': response.text[:200]}
                
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def create_vertex_ai_lambda(self, role_arn: str) -> Dict:
        """Create Lambda function that uses Vertex AI"""
        print("\n[2/6] Creating Vertex AI Lambda Function...")
        
        function_name = 'asi-vertex-ai-chat'
        
        lambda_code = f'''
import json
import requests

VERTEX_API_KEY = '{self.vertex_api_key}'
VERTEX_ENDPOINT = '{self.vertex_endpoint}'

def lambda_handler(event, context):
    """Vertex AI chat endpoint"""
    
    try:
        # Parse request
        body = json.loads(event.get('body', '{{}}'))
        prompt = body.get('prompt', 'Hello')
        
        # Call Vertex AI
        response = requests.post(
            f"{{VERTEX_ENDPOINT}}?key={{VERTEX_API_KEY}}",
            headers={{'Content-Type': 'application/json'}},
            json={{
                'contents': [{{
                    'role': 'user',
                    'parts': [{{'text': prompt}}]
                }}]
            }},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()[0]
            text = result['candidates'][0]['content']['parts'][0]['text']
            model = result['modelVersion']
            
            return {{
                'statusCode': 200,
                'headers': {{
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }},
                'body': json.dumps({{
                    'response': text,
                    'model': model,
                    'provider': 'vertex_ai'
                }})
            }}
        else:
            return {{
                'statusCode': 500,
                'body': json.dumps({{'error': 'Vertex AI error'}})
            }}
            
    except Exception as e:
        return {{
            'statusCode': 500,
            'body': json.dumps({{'error': str(e)}})
        }}
'''
        
        # Create deployment package
        zip_path = '/tmp/vertex_ai_chat.zip'
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('lambda_function.py', lambda_code)
        
        with open(zip_path, 'rb') as f:
            zip_content = f.read()
        
        try:
            response = self.lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_content
            )
            print(f"   ✅ Updated existing function")
            function_arn = response['FunctionArn']
        except:
            try:
                response = self.lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime='python3.11',
                    Role=role_arn,
                    Handler='lambda_function.lambda_handler',
                    Code={'ZipFile': zip_content},
                    Timeout=30,
                    MemorySize=512,
                    Description='ASI Vertex AI Chat Endpoint',
                    # Layers removed - requests is built-in
                )
                function_arn = response['FunctionArn']
                print(f"   ✅ Created new function: {function_arn}")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                return {'status': 'FAIL', 'error': str(e)}
        
        return {'status': 'SUCCESS', 'arn': function_arn}
    
    def create_agent_executor_lambda(self, role_arn: str) -> Dict:
        """Create agent executor Lambda function"""
        print("\n[3/6] Creating Agent Executor Lambda...")
        
        function_name = 'asi-agent-executor'
        
        lambda_code = f'''
import json
import boto3
import requests
from datetime import datetime

dynamodb = boto3.resource('dynamodb')
VERTEX_API_KEY = '{self.vertex_api_key}'
VERTEX_ENDPOINT = '{self.vertex_endpoint}'

def lambda_handler(event, context):
    """Execute agent tasks"""
    
    try:
        # Parse task
        task = json.loads(event.get('body', '{{}}'))
        task_id = task.get('id', context.request_id)
        task_type = task.get('type', 'general')
        task_prompt = task.get('prompt', 'Process this task')
        
        # Call Vertex AI to process task
        response = requests.post(
            f"{{VERTEX_ENDPOINT}}?key={{VERTEX_API_KEY}}",
            headers={{'Content-Type': 'application/json'}},
            json={{
                'contents': [{{
                    'role': 'user',
                    'parts': [{{'text': f"Task: {{task_prompt}}"}}]
                }}]
            }},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()[0]
            ai_response = result['candidates'][0]['content']['parts'][0]['text']
            
            # Save result to DynamoDB
            table = dynamodb.Table('multi-agent-asi-system')
            table.put_item(
                Item={{
                    'id': task_id,
                    'type': 'task_result',
                    'task_type': task_type,
                    'result': ai_response,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed'
                }}
            )
            
            return {{
                'statusCode': 200,
                'headers': {{
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }},
                'body': json.dumps({{
                    'task_id': task_id,
                    'status': 'completed',
                    'result': ai_response
                }})
            }}
        else:
            return {{
                'statusCode': 500,
                'body': json.dumps({{'error': 'AI processing failed'}})
            }}
            
    except Exception as e:
        return {{
            'statusCode': 500,
            'body': json.dumps({{'error': str(e)}})
        }}
'''
        
        zip_path = '/tmp/agent_executor.zip'
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('lambda_function.py', lambda_code)
        
        with open(zip_path, 'rb') as f:
            zip_content = f.read()
        
        try:
            response = self.lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_content
            )
            print(f"   ✅ Updated existing function")
            function_arn = response['FunctionArn']
        except:
            try:
                response = self.lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime='python3.11',
                    Role=role_arn,
                    Handler='lambda_function.lambda_handler',
                    Code={'ZipFile': zip_content},
                    Timeout=60,
                    MemorySize=1024,
                    Description='ASI Agent Task Executor',
                    # Layers removed - requests is built-in
                )
                function_arn = response['FunctionArn']
                print(f"   ✅ Created new function: {function_arn}")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                return {'status': 'FAIL', 'error': str(e)}
        
        return {'status': 'SUCCESS', 'arn': function_arn}
    
    def create_function_urls(self) -> Dict:
        """Create function URLs for new Lambda functions"""
        print("\n[4/6] Creating Function URLs...")
        
        urls = {}
        
        for function_name in ['asi-vertex-ai-chat', 'asi-agent-executor']:
            try:
                response = self.lambda_client.create_function_url_config(
                    FunctionName=function_name,
                    AuthType='NONE',
                    Cors={
                        'AllowOrigins': ['*'],
                        'AllowMethods': ['GET', 'POST'],
                        'AllowHeaders': ['*'],
                        'MaxAge': 86400
                    }
                )
                url = response['FunctionUrl']
                
                # Add permission
                self.lambda_client.add_permission(
                    FunctionName=function_name,
                    StatementId='FunctionURLAllowPublicAccess',
                    Action='lambda:InvokeFunctionUrl',
                    Principal='*',
                    FunctionUrlAuthType='NONE'
                )
                
                urls[function_name] = url
                print(f"   ✅ {function_name}: {url}")
            except self.lambda_client.exceptions.ResourceConflictException:
                response = self.lambda_client.get_function_url_config(
                    FunctionName=function_name
                )
                url = response['FunctionUrl']
                urls[function_name] = url
                print(f"   ✅ {function_name}: {url} (existing)")
            except Exception as e:
                print(f"   ⚠️  {function_name}: {str(e)}")
        
        return urls
    
    def test_deployed_functions(self, urls: Dict) -> List:
        """Test deployed functions"""
        print("\n[5/6] Testing Deployed Functions...")
        
        tests = []
        
        # Test Vertex AI chat
        if 'asi-vertex-ai-chat' in urls:
            try:
                import time
                time.sleep(2)  # Wait for function to be ready
                
                response = requests.post(
                    urls['asi-vertex-ai-chat'],
                    json={'prompt': 'Say: TEST SUCCESSFUL'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Vertex AI Chat: {result.get('response', '')[:50]}...")
                    tests.append({'function': 'vertex-ai-chat', 'status': 'PASS'})
                else:
                    print(f"   ❌ Vertex AI Chat: Status {response.status_code}")
                    tests.append({'function': 'vertex-ai-chat', 'status': 'FAIL'})
            except Exception as e:
                print(f"   ⚠️  Vertex AI Chat: {str(e)}")
                tests.append({'function': 'vertex-ai-chat', 'status': 'ERROR', 'error': str(e)})
        
        # Test Agent Executor
        if 'asi-agent-executor' in urls:
            try:
                import time
                time.sleep(2)
                
                response = requests.post(
                    urls['asi-agent-executor'],
                    json={
                        'id': 'test-001',
                        'type': 'test',
                        'prompt': 'Calculate 2+2 and explain'
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Agent Executor: Task {result.get('status', '')}")
                    tests.append({'function': 'agent-executor', 'status': 'PASS'})
                else:
                    print(f"   ❌ Agent Executor: Status {response.status_code}")
                    tests.append({'function': 'agent-executor', 'status': 'FAIL'})
            except Exception as e:
                print(f"   ⚠️  Agent Executor: {str(e)}")
                tests.append({'function': 'agent-executor', 'status': 'ERROR', 'error': str(e)})
        
        return tests
    
    def execute_phase3(self):
        """Execute Phase 3"""
        print("="*80)
        print("PHASE 3: VERTEX AI INTEGRATION + AGENT DEPLOYMENT")
        print("Goal: 62/100 → 85/100")
        print("="*80)
        
        # Get IAM role
        iam = boto3.client('iam')
        role = iam.get_role(RoleName='ASI-Lambda-Execution-Role')
        role_arn = role['Role']['Arn']
        
        # Test Vertex AI
        vertex_test = self.test_vertex_ai()
        
        # Deploy Lambda functions
        vertex_lambda = self.create_vertex_ai_lambda(role_arn)
        agent_lambda = self.create_agent_executor_lambda(role_arn)
        
        # Create function URLs
        urls = self.create_function_urls()
        
        # Test functions
        print("\n[6/6] Waiting for functions to be ready...")
        import time
        time.sleep(5)
        tests = self.test_deployed_functions(urls)
        
        # Calculate score
        points = 0
        if vertex_test.get('status') == 'PASS':
            points += 8  # Vertex AI working
        if vertex_lambda.get('status') == 'SUCCESS':
            points += 5  # Vertex Lambda deployed
        if agent_lambda.get('status') == 'SUCCESS':
            points += 5  # Agent Lambda deployed
        points += len(urls) * 2  # URLs created
        points += len([t for t in tests if t.get('status') == 'PASS']) * 2  # Tests passed
        
        self.results['score_after'] = 62 + points
        self.results['urls'] = urls
        self.results['tests'] = tests
        
        # Save results
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"PHASE3/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            Body=json.dumps(self.results, indent=2)
        )
        
        print("\n" + "="*80)
        print("PHASE 3 COMPLETE")
        print("="*80)
        print(f"\n✅ Vertex AI: {vertex_test.get('status')}")
        print(f"\n🌐 Function URLs:")
        for name, url in urls.items():
            print(f"   - {name}: {url}")
        print(f"\n📊 Score: {self.results['score_before']}/100 → {self.results['score_after']}/100")
        print(f"   Progress: +{self.results['score_after'] - self.results['score_before']} points")
        print("\n✅ Results saved to S3")
        print("="*80)
        
        return self.results

def main():
    phase3 = Phase3VertexAIAgents()
    phase3.execute_phase3()

if __name__ == '__main__':
    main()
