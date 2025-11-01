# TRUE ASI System - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Clone the Repository

```bash
git clone https://github.com/AICSSUPERVISOR/true-asi-system.git
cd true-asi-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and AWS credentials
```

### 4. Run the ASI Engine

```bash
python -m src.core.asi_engine
```

### 5. Test the System

```bash
pytest tests/ -v
```

## 📊 Verify Installation

```bash
python scripts/verify_system.py
```

Expected output:
```
✅ Python version: OK
✅ Directory src: OK
✅ Directory agents: OK
✅ Agents: OK (250/250)
✅ System verification complete
```

## 🎯 Quick Examples

### Example 1: Initialize ASI Engine

```python
from src.core.asi_engine import ASIEngine

# Create ASI instance
asi = ASIEngine()

# Process a task
result = await asi.process_task({
    'type': 'reasoning',
    'query': 'Analyze the implications of quantum computing on AI',
    'context': {}
})

print(result)
```

### Example 2: Use Agent Manager

```python
from src.agents.agent_manager import AgentManager

# Initialize agents
manager = AgentManager()
await manager.initialize_agents(250)

# Assign a task
result = await manager.assign_task({
    'type': 'data_processing',
    'data': 'Sample data to process'
})

print(result)
```

### Example 3: Query Knowledge Graph

```python
from src.knowledge.knowledge_graph import KnowledgeGraph

# Create knowledge graph
kg = KnowledgeGraph()
await kg.initialize()

# Add entity
await kg.add_entity({
    'name': 'Quantum Computing',
    'type': 'Technology',
    'description': 'Computing using quantum-mechanical phenomena'
})

# Query
results = await kg.query('quantum')
print(results)
```

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

### Stop Services

```bash
docker-compose down
```

## ☸️ Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods

# View logs
kubectl logs -f <pod-name>
```

## 🔧 Troubleshooting

### Issue: AWS Credentials Not Found

**Solution**: Ensure `.env` file contains valid AWS credentials:
```bash
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=us-east-1
```

### Issue: OpenAI API Error

**Solution**: Verify your OpenAI API key in `.env`:
```bash
OPENAI_API_KEY=sk-...
```

### Issue: Agent Initialization Fails

**Solution**: Ensure all 250 agent files exist:
```bash
ls agents/ | wc -l  # Should output 250
```

## 📚 Next Steps

1. **Read Documentation**: Check `docs/` directory for detailed guides
2. **Explore Agents**: Review `agents/` directory to understand agent specialties
3. **Run Tests**: Execute `pytest tests/` to verify functionality
4. **Deploy to AWS**: Follow `docs/DEPLOYMENT.md` for production deployment
5. **Contribute**: See `docs/CONTRIBUTING.md` for contribution guidelines

## 🆘 Get Help

- **Documentation**: `docs/`
- **GitHub Issues**: https://github.com/AICSSUPERVISOR/true-asi-system/issues
- **Playbook**: `docs/PLAYBOOK.md`
- **Metrics**: `docs/METRICS.md`

## ⚡ Performance Tips

1. **Use Multiple Workers**: Set `MAX_WORKERS=20` in `.env`
2. **Batch Processing**: Set `BATCH_SIZE=500` for optimal throughput
3. **Enable Caching**: Use DynamoDB for entity caching
4. **Monitor Metrics**: Check `docs/METRICS.md` for real-time stats

---

**Ready to build TRUE ASI?** 🚀🧠✨

*For detailed documentation, see `README.md` and `docs/ARCHITECTURE.md`*
