"""
PHASE 26: FIX AGENT ORCHESTRATOR
Goal: Debug and fix the 500 error in the Agent Orchestrator API
"""

import json
import boto3
import time
import requests
from datetime import datetime

class Phase26FixOrchestrator:
    def __init__(self):
        self.lambda_client = boto3.client("lambda")
        self.s3 = boto3.client("s3")
        self.bucket = "asi-knowledge-base-898982995956"
        self.function_name = "asi-agent-orchestrator"
        self.api_url = "https://5w3sf4a3urxhj73iuf6cotw3jm0nwzkk.lambda-url.us-east-1.on.aws/"

    def fix_and_deploy(self):
        print("\n" + "=" * 80)
        print("PHASE 26: FIXING AGENT ORCHESTRATOR")
        print("=" * 80 + "\n")

        try:
            response = self.lambda_client.get_function_configuration(FunctionName=self.function_name)
            role_arn = response["Role"]
            role_name = role_arn.split("/")[-1]

            iam = boto3.client("iam")

            print(f"Attaching SQS permissions to role: {role_name}")
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSQSFullAccess",
            )

            print(f"Attaching DynamoDB permissions to role: {role_name}")
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonDynamoDBReadOnlyAccess",
            )

            print("✅ Permissions attached successfully.")
            time.sleep(10)

        except Exception as e:
            print(f"❌ Error attaching permissions: {e}")
            return False

        print("\nTesting API after permission fix...")
        try:
            r = requests.post(self.api_url, json={"task": "test"}, timeout=15)
            if r.status_code == 200:
                print("✅ SUCCESS! Agent Orchestrator is now working (200)")
                return True
            else:
                print(f"❌ FAILED! Still getting status {r.status_code}")
                print(r.text)
                return False
        except Exception as e:
            print(f"❌ ERROR testing API: {e}")
            return False

    def run_phase26(self):
        success = self.fix_and_deploy()

        result = {
            "phase": 26,
            "name": "Fix Agent Orchestrator",
            "timestamp": datetime.now().isoformat(),
            "success": success,
        }
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"PHASE26_RESULTS/results_{date_str}.json"

        self.s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=json.dumps(result, indent=2),
        )

        print("\n" + "=" * 80)
        print("PHASE 26 COMPLETE")
        status_str = "SUCCESS" if success else "FAILED"
        print(f"Status: {status_str}")
        print("Results saved to S3.")
        print("=" * 80)

if __name__ == "__main__":
    phase26 = Phase26FixOrchestrator()
    phase26.run_phase26()
