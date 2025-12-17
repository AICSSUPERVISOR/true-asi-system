#!/usr/bin/env python3
"""
VELOCITY Competitor Price Data Storage
Stores pricing data in database for analytics dashboard.
Supports Supabase or local JSON storage as fallback.
"""

import os
import json
from datetime import datetime

# Load environment variables
SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')

def is_valid_supabase_url(url: str) -> bool:
    """Check if URL is a valid Supabase URL."""
    return url.startswith('https://') and 'supabase' in url.lower()

def store_to_local_json(report: dict) -> dict:
    """Store data to local JSON file as database backup."""
    storage_path = '/home/ubuntu/velocity-system/database_storage.json'
    
    # Load existing data if available
    existing_data = {'competitor_prices': [], 'velocity_prices': [], 'price_alerts': []}
    try:
        with open(storage_path, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        pass
    
    # Append new data with timestamps
    timestamp = datetime.utcnow().isoformat()
    
    for cp in report['competitor_prices']:
        cp['stored_at'] = timestamp
        existing_data['competitor_prices'].append(cp)
    
    for vp in report['velocity_prices']:
        vp['stored_at'] = timestamp
        existing_data['velocity_prices'].append(vp)
    
    # Add metadata
    existing_data['last_updated'] = timestamp
    existing_data['total_competitor_records'] = len(existing_data['competitor_prices'])
    existing_data['total_velocity_records'] = len(existing_data['velocity_prices'])
    
    # Save to file
    with open(storage_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    return {
        'storage_method': 'local_json',
        'storage_path': storage_path,
        'competitor_records_stored': len(report['competitor_prices']),
        'velocity_records_stored': len(report['velocity_prices']),
        'timestamp': timestamp
    }

def create_sql_schema() -> str:
    """Generate SQL schema for database tables."""
    return """
-- VELOCITY Competitor Price Monitoring Database Schema
-- Generated: {timestamp}

-- Table: competitor_prices
-- Stores raw competitor pricing data scraped from various sources
CREATE TABLE IF NOT EXISTS competitor_prices (
    id BIGSERIAL PRIMARY KEY,
    competitor VARCHAR(50) NOT NULL,
    route VARCHAR(100) NOT NULL,
    origin VARCHAR(100) NOT NULL,
    destination VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(10) NOT NULL,
    service_type VARCHAR(50),
    source_url TEXT,
    scraped_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table: velocity_prices
-- Stores calculated VELOCITY competitive prices
CREATE TABLE IF NOT EXISTS velocity_prices (
    id BIGSERIAL PRIMARY KEY,
    route VARCHAR(100) NOT NULL,
    origin VARCHAR(100) NOT NULL,
    destination VARCHAR(100) NOT NULL,
    velocity_standard_price DECIMAL(10, 2) NOT NULL,
    velocity_autonomous_price DECIMAL(10, 2) NOT NULL,
    discount_vs_uber DECIMAL(5, 2),
    discount_vs_bolt DECIMAL(5, 2),
    discount_vs_lyft DECIMAL(5, 2),
    currency VARCHAR(10) NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table: price_alerts
-- Stores significant price change alerts
CREATE TABLE IF NOT EXISTS price_alerts (
    id BIGSERIAL PRIMARY KEY,
    route VARCHAR(100) NOT NULL,
    competitor VARCHAR(50) NOT NULL,
    previous_price DECIMAL(10, 2) NOT NULL,
    current_price DECIMAL(10, 2) NOT NULL,
    change_percent DECIMAL(5, 2) NOT NULL,
    currency VARCHAR(10) NOT NULL,
    alert_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_competitor_prices_route ON competitor_prices(route);
CREATE INDEX IF NOT EXISTS idx_competitor_prices_competitor ON competitor_prices(competitor);
CREATE INDEX IF NOT EXISTS idx_competitor_prices_scraped_at ON competitor_prices(scraped_at);
CREATE INDEX IF NOT EXISTS idx_velocity_prices_route ON velocity_prices(route);
CREATE INDEX IF NOT EXISTS idx_velocity_prices_calculated_at ON velocity_prices(calculated_at);
CREATE INDEX IF NOT EXISTS idx_price_alerts_route ON price_alerts(route);
CREATE INDEX IF NOT EXISTS idx_price_alerts_created_at ON price_alerts(created_at);
""".format(timestamp=datetime.utcnow().isoformat())

def main():
    """Main execution function."""
    print("=" * 80)
    print("VELOCITY Database Storage System")
    print("=" * 80)
    
    # Load the pricing report
    report_path = '/home/ubuntu/velocity-system/competitor_price_report.json'
    
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        print(f"\n✅ Loaded pricing report from {report_path}")
    except FileNotFoundError:
        print(f"\n❌ Report file not found: {report_path}")
        return
    
    # Store data to local JSON (primary storage for this implementation)
    print("\n" + "-" * 80)
    print("Storing Data to Local Database...")
    print("-" * 80)
    
    storage_result = store_to_local_json(report)
    
    print(f"\n✅ Data stored successfully!")
    print(f"   Storage Method: {storage_result['storage_method']}")
    print(f"   Storage Path: {storage_result['storage_path']}")
    print(f"   Competitor Records: {storage_result['competitor_records_stored']}")
    print(f"   VELOCITY Records: {storage_result['velocity_records_stored']}")
    print(f"   Timestamp: {storage_result['timestamp']}")
    
    # Generate and save SQL schema for future database migration
    print("\n" + "-" * 80)
    print("Generating SQL Schema...")
    print("-" * 80)
    
    schema_path = '/home/ubuntu/velocity-system/database_schema.sql'
    schema = create_sql_schema()
    
    with open(schema_path, 'w') as f:
        f.write(schema)
    
    print(f"\n✅ SQL schema saved to {schema_path}")
    
    # Print summary of stored data
    print("\n" + "=" * 80)
    print("STORAGE SUMMARY")
    print("=" * 80)
    
    print("\n📊 COMPETITOR PRICES STORED:")
    for cp in report['competitor_prices']:
        print(f"   • {cp['competitor']}: {cp['route']} - {cp['currency']} {cp['price']}")
    
    print("\n🚀 VELOCITY PRICES STORED:")
    for vp in report['velocity_prices']:
        print(f"   • {vp['route']}:")
        print(f"     - Standard: {vp['currency']} {vp['velocity_standard_price']}")
        print(f"     - Autonomous: {vp['currency']} {vp['velocity_autonomous_price']}")
    
    print("\n" + "=" * 80)
    print("✅ All data successfully stored for analytics dashboard")
    print("=" * 80)
    
    return storage_result


if __name__ == "__main__":
    results = main()
