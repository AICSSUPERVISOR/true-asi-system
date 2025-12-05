# Redis Deployment Guide

This guide provides step-by-step instructions for deploying Redis in production to enable caching for Forvalt.no credit rating lookups.

## Why Redis?

Redis caching provides significant performance improvements:
- **Without cache:** 10-30 seconds (Puppeteer web scraping)
- **With cache:** <100ms (Redis lookup)
- **Expected cache hit rate:** 80%+
- **Load reduction:** 80%+ fewer requests to Forvalt.no

## Option 1: Upstash (Recommended for Serverless)

**Pros:**
- Serverless Redis (pay-per-request)
- Global edge caching
- Free tier: 10,000 commands/day
- No server management
- Built-in TLS encryption
- REST API available

**Setup Steps:**

1. Go to [https://upstash.com](https://upstash.com)
2. Sign up / Log in
3. Click "Create Database"
4. Choose:
   - **Name:** true-asi-redis
   - **Type:** Regional
   - **Region:** Europe (closest to Norway)
   - **TLS:** Enabled
5. Copy connection details:
   ```
   UPSTASH_REDIS_REST_URL=https://xxx.upstash.io
   UPSTASH_REDIS_REST_TOKEN=xxx
   ```
6. Add to Manus environment variables:
   ```
   REDIS_HOST=xxx.upstash.io
   REDIS_PORT=6379
   REDIS_PASSWORD=xxx
   ```

**Pricing:**
- Free: 10,000 commands/day, 256MB storage
- Pay-as-you-go: $0.2 per 100K commands
- Pro: $280/month (unlimited)

---

## Option 2: Railway (Recommended for Simplicity)

**Pros:**
- Simple deployment
- Free $5 credit/month
- Automatic backups
- Built-in monitoring
- One-click deploy

**Setup Steps:**

1. Go to [https://railway.app](https://railway.app)
2. Sign up / Log in with GitHub
3. Click "New Project"
4. Click "Deploy Redis"
5. Wait for deployment (1-2 minutes)
6. Click on Redis service
7. Go to "Variables" tab
8. Copy connection details:
   ```
   REDIS_HOST=containers-us-west-xxx.railway.app
   REDIS_PORT=6379
   REDIS_PASSWORD=xxx
   ```
9. Add to Manus environment variables

**Pricing:**
- Free: $5 credit/month (~500 hours)
- Hobby: $5/month
- Pro: $20/month

---

## Option 3: AWS ElastiCache (Recommended for Enterprise)

**Pros:**
- Fully managed Redis
- High availability
- Automatic failover
- VPC security
- CloudWatch monitoring
- Backup and restore

**Setup Steps:**

1. Go to AWS Console → ElastiCache
2. Click "Create"
3. Choose:
   - **Engine:** Redis
   - **Version:** 7.0
   - **Node type:** cache.t3.micro (free tier)
   - **Number of replicas:** 0 (or 1 for HA)
   - **Multi-AZ:** Disabled (or enabled for HA)
4. Configure security group:
   - Allow inbound port 6379 from your app
5. Wait for deployment (10-15 minutes)
6. Copy connection details:
   ```
   REDIS_HOST=xxx.cache.amazonaws.com
   REDIS_PORT=6379
   REDIS_PASSWORD=xxx (if auth enabled)
   ```
7. Add to Manus environment variables

**Pricing:**
- Free tier: cache.t3.micro (750 hours/month for 12 months)
- cache.t3.micro: $0.017/hour (~$12/month)
- cache.t3.small: $0.034/hour (~$25/month)
- cache.t3.medium: $0.068/hour (~$50/month)

---

## Option 4: Redis Cloud (Managed Redis by Redis Labs)

**Pros:**
- Official Redis hosting
- Free tier: 30MB
- Global replication
- Active-active geo-distribution
- Redis modules (JSON, Search, etc.)

**Setup Steps:**

1. Go to [https://redis.com/try-free/](https://redis.com/try-free/)
2. Sign up / Log in
3. Click "New Subscription"
4. Choose:
   - **Cloud:** AWS
   - **Region:** Europe (eu-west-1)
   - **Plan:** Free (30MB)
5. Click "Create Database"
6. Copy connection details:
   ```
   REDIS_HOST=redis-xxx.cloud.redislabs.com
   REDIS_PORT=6379
   REDIS_PASSWORD=xxx
   ```
7. Add to Manus environment variables

**Pricing:**
- Free: 30MB storage
- Essentials: $7/month (250MB)
- Pro: Custom pricing

---

## Configuration in Manus

After deploying Redis, add these environment variables in Manus Settings → Secrets:

```bash
REDIS_HOST=your-redis-host.com
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
```

**Note:** The TRUE ASI system already has Redis caching implemented. It will automatically use these credentials when available, and gracefully fall back to no caching if Redis is unavailable.

---

## Verification

After adding environment variables, restart the dev server and check logs:

```bash
# Success:
[Redis] Connected successfully

# Failure (graceful fallback):
[Redis] Connection error (cache disabled): ECONNREFUSED
```

Test caching with a Forvalt.no lookup:

```bash
# First request (cache miss):
[Forvalt] Cache miss for 923609016, scraping...
[Forvalt] Cached data for 923609016

# Second request (cache hit):
[Forvalt] Cache hit for 923609016
```

---

## Cache Management

### Clear all Forvalt.no cache:
```typescript
import { invalidateAllForvaltCache } from './server/helpers/redis_cache';
await invalidateAllForvaltCache();
```

### Clear specific company cache:
```typescript
import { invalidateForvaltCache } from './server/helpers/redis_cache';
await invalidateForvaltCache('923609016');
```

### Adjust TTL (Time To Live):
Edit `server/helpers/forvalt_scraper.ts`:
```typescript
// Change from 24 hours to 12 hours
await setCachedForvaltData(orgNumber, forvaltData, 43200); // 12 hours in seconds
```

---

## Monitoring

### Upstash:
- Dashboard: https://console.upstash.com
- Metrics: Commands, latency, storage

### Railway:
- Dashboard: https://railway.app
- Metrics: CPU, memory, network

### AWS ElastiCache:
- CloudWatch: CPU, memory, cache hits/misses
- Alarms: Set up alerts for high CPU/memory

### Redis Cloud:
- Dashboard: https://app.redislabs.com
- Metrics: Operations/sec, latency, memory

---

## Troubleshooting

### Connection refused:
- Check firewall/security group allows port 6379
- Verify REDIS_HOST is correct
- Check Redis service is running

### Authentication failed:
- Verify REDIS_PASSWORD is correct
- Check if Redis requires TLS (use `rediss://` URL)

### High memory usage:
- Check cache size: `redis-cli INFO memory`
- Clear old keys: `redis-cli FLUSHDB`
- Reduce TTL in code

### Slow performance:
- Check Redis latency: `redis-cli --latency`
- Move Redis closer to app (same region)
- Upgrade Redis instance size

---

## Recommended Setup

For TRUE ASI production deployment:

1. **Development:** No Redis (graceful fallback)
2. **Staging:** Railway ($5/month)
3. **Production:** Upstash Pro or AWS ElastiCache (high availability)

**Production Configuration:**
- **Provider:** Upstash or AWS ElastiCache
- **Region:** Europe (eu-west-1 or eu-north-1)
- **Size:** 1GB memory
- **Replication:** Enabled (1 replica)
- **Backup:** Daily automated backups
- **Monitoring:** CloudWatch or Upstash dashboard
- **Estimated cost:** $20-50/month

---

## Next Steps

1. Choose Redis provider (Upstash recommended for simplicity)
2. Create Redis instance
3. Copy connection credentials
4. Add to Manus environment variables
5. Restart dev server
6. Test Forvalt.no caching
7. Monitor cache hit rate
8. Enjoy 80%+ performance improvement! 🚀
