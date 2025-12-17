#!/usr/bin/env python3.11
"""
POST-PHASE 16 BRUTAL AUDIT
Test all 8 APIs after production hardening
"""

import requests
import json
import boto3
from datetime import datetime

class PostPhase16Audit:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.bucket = 'asi-knowledge-base-898982995956'
        
        self.apis = {
            'health': 'https://am3q7njcihyeqqkwb67s6yhbhy0ldcfy.lambda-url.us-east-1.on.aws/',
            'models': 'https://4fukiyti7tdhdm4aercavqunwe0nxtlj.lambda-url.us-east-1.on.aws/',
            'chat': 'https://iiasi5ibfhehfjcb66alny66vm0gledr.lambda-url.us-east-1.on.aws/',
            'agent': 'https://t3j2tgdaxsrpofpnt3evkwihzy0zbczm.lambda-url.us-east-1.on.aws/',
            'router': 'https://vfg2sio7mjoodafkwtzkpp4yu40dqvex.lambda-url.us-east-1.on.aws/',
            'orchestrator': 'https://5w3sf4a3urxhj73iuf6cotw3jm0nwzkk.lambda-url.us-east-1.on.aws/',
            'knowledge': 'https://5ukzohy5jde4u2mmzln62pb2va0rfgkf.lambda-url.us-east-1.on.aws/',
            'reasoning': 'https://jenw2ecbs3fq2gjjbbz4soywg40mckns.lambda-url.us-east-1.on.aws/'
        }
    
    def test_all_apis(self):
        print("\n" + "="*80)
        print("POST-PHASE 16 BRUTAL AUDIT")
        print("="*80 + "\n")
        
        results = {}
        for name, url in self.apis.items():
            try:
                if name in ['agent', 'chat', 'router', 'orchestrator', 'knowledge', 'reasoning']:
                    r = requests.post(url, json={'prompt': 'test'}, timeout=10)
                else:
                    r = requests.get(url, timeout=10)
                
                status = "✅" if r.status_code == 200 else "❌"
                print(f"{status} {name}: {r.status_code}")
                results[name] = {'status': r.status_code, 'success': r.status_code == 200}
                
            except Exception as e:
                print(f"❌ {name}: ERROR - {str(e)[:50]}")
                results[name] = {'status': 'ERROR', 'success': False}
        
        # Calculate score
        successful = sum(1 for r in results.values() if r.get('success', False))
        score = (successful / len(self.apis)) * 100
        
        print(f"\n{'='*80}")
        print(f"SCORE: {successful}/{len(self.apis)} APIs working ({score:.1f}%)")
        print(f"{'='*80}\n")
        
        # Save to S3
        audit_result = {
            'phase': 'post-16',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'score': score,
            'successful': successful,
            'total': len(self.apis)
        }
        
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"BRUTAL_AUDITS/post_phase16_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            Body=json.dumps(audit_result, indent=2)
        )
        
        return audit_result

def main():
    auditor = PostPhase16Audit()
    result = auditor.test_all_apis()
    
    print(f"✅ Audit complete. Score: {result['score']:.1f}%")
    print(f"Saved to S3: s3://asi-knowledge-base-898982995956/BRUTAL_AUDITS/")

if __name__ == '__main__':
    main()
