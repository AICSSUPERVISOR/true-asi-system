#!/usr/bin/env python3.11
"""
20 CYCLES OF ICE COLD BRUTAL AUDITS
Non-destructive comprehensive testing to uncover ALL gaps
"""

import json
import boto3
import requests
import time
from datetime import datetime
from typing import Dict, List

class TwentyCycleBrutalAudit:
    """Run 20 cycles of comprehensive brutal audits"""
    
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
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
        
        self.all_cycles = []
        self.gaps_found = set()
    
    def single_cycle_audit(self, cycle_num: int) -> Dict:
        """Run a single comprehensive audit cycle"""
        print(f"\n{'='*80}")
        print(f"CYCLE {cycle_num}/20 - BRUTAL AUDIT")
        print(f"{'='*80}\n")
        
        results = {
            'cycle': cycle_num,
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Test all APIs
        print(f"[{cycle_num}] Testing All APIs...")
        for name, url in self.apis.items():
            try:
                if name in ['agent', 'chat', 'router', 'orchestrator', 'knowledge', 'reasoning']:
                    r = requests.post(url, json={'prompt': 'test'}, timeout=5)
                else:
                    r = requests.get(url, timeout=5)
                
                results['tests'][name] = {
                    'status': r.status_code,
                    'success': r.status_code == 200,
                    'latency_ms': r.elapsed.total_seconds() * 1000
                }
                
                if r.status_code != 200:
                    self.gaps_found.add(f"{name}_api_error_{r.status_code}")
                
            except Exception as e:
                results['tests'][name] = {
                    'status': 'ERROR',
                    'success': False,
                    'error': str(e)[:100]
                }
                self.gaps_found.add(f"{name}_api_timeout_or_error")
        
        # Calculate cycle score
        successful = sum(1 for t in results['tests'].values() if t.get('success', False))
        results['score'] = (successful / len(self.apis)) * 100
        results['successful'] = successful
        results['total'] = len(self.apis)
        
        return results
    
    def run_20_cycles(self):
        """Run 20 complete audit cycles"""
        print("\n" + "="*80)
        print("STARTING 20 CYCLES OF BRUTAL AUDITS")
        print("="*80)
        
        for cycle in range(1, 21):
            cycle_result = self.single_cycle_audit(cycle)
            self.all_cycles.append(cycle_result)
            
            print(f"   Cycle {cycle}: {cycle_result['successful']}/{cycle_result['total']} APIs working ({cycle_result['score']:.1f}%)")
            
            # Small delay between cycles
            if cycle < 20:
                time.sleep(2)
        
        # Analyze results
        print("\n" + "="*80)
        print("20-CYCLE AUDIT COMPLETE - ANALYSIS")
        print("="*80 + "\n")
        
        # Calculate statistics
        scores = [c['score'] for c in self.all_cycles]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # API reliability
        api_success_rates = {}
        for api_name in self.apis.keys():
            successes = sum(1 for c in self.all_cycles if c['tests'].get(api_name, {}).get('success', False))
            api_success_rates[api_name] = (successes / 20) * 100
        
        analysis = {
            'total_cycles': 20,
            'average_score': round(avg_score, 1),
            'min_score': round(min_score, 1),
            'max_score': round(max_score, 1),
            'api_reliability': api_success_rates,
            'gaps_found': list(self.gaps_found),
            'all_cycles': self.all_cycles
        }
        
        print(f"Average Score: {avg_score:.1f}%")
        print(f"Score Range: {min_score:.1f}% - {max_score:.1f}%")
        print(f"\nAPI Reliability (% uptime across 20 cycles):")
        for api, rate in sorted(api_success_rates.items(), key=lambda x: x[1], reverse=True):
            status = "✅" if rate >= 95 else ("⚠️" if rate >= 80 else "❌")
            print(f"   {status} {api}: {rate:.0f}%")
        
        print(f"\nTotal Gaps Found: {len(self.gaps_found)}")
        for gap in sorted(self.gaps_found):
            print(f"   - {gap}")
        
        # Save to S3
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"20_CYCLE_AUDIT/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            Body=json.dumps(analysis, indent=2)
        )
        
        print(f"\n✅ All results saved to S3: s3://{self.bucket}/20_CYCLE_AUDIT/")
        
        return analysis

def main():
    auditor = TwentyCycleBrutalAudit()
    results = auditor.run_20_cycles()
    
    print("\n" + "="*80)
    print("20-CYCLE BRUTAL AUDIT COMPLETE")
    print(f"Ready for web research phase to identify additional ASI requirements")
    print("="*80)

if __name__ == '__main__':
    main()
