
-- VELOCITY Competitor Price Monitoring Database Schema
-- Generated: 2025-12-17T23:06:29.474949

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
