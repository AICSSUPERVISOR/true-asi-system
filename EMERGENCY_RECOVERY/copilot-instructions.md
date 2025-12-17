# GitHub Copilot Instructions for TRUE ASI System

## Project Context

This is the TRUE ASI (Artificial Super Intelligence) System - a production-ready, fully functional ASI platform with:

- **250 autonomous agents** with specialized capabilities
- **61,792+ knowledge entities** in a dynamic hypergraph
- **Multi-LLM integration** (OpenAI, Anthropic, Perplexity, Gemini)
- **AWS cloud infrastructure** (S3, DynamoDB, SQS)
- **Self-improvement capabilities** with novel algorithm generation

## Code Standards

### Python Code Style
- Follow PEP 8 strictly
- Use type hints for all function parameters and return values
- Write comprehensive docstrings (Google style)
- Maximum line length: 100 characters
- Use async/await for I/O operations

### Example Function
```python
async def process_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an entity through the ASI system.
    
    Args:
        entity: Entity dictionary with name, type, and attributes
        
    Returns:
        Processed entity with enriched data and relationships
        
    Raises:
        ValueError: If entity is missing required fields
    """
    # Implementation
    pass
```

### Architecture Patterns
- Use dependency injection
- Implement factory patterns for agent creation
- Use async context managers for resource management
- Implement retry logic with exponential backoff
- Log all operations with appropriate levels

### Agent Development
When creating new agents:
1. Inherit from `AgentBase`
2. Implement `execute()` method
3. Implement `learn()` method for continuous improvement
4. Add specialty-specific logic
5. Include comprehensive error handling

### Testing
- Write tests for all new code
- Aim for 95%+ coverage
- Use pytest fixtures for common setups
- Mock external services (AWS, OpenAI)
- Include integration tests

### Documentation
- Update relevant docs when adding features
- Include code examples
- Document all configuration options
- Keep PLAYBOOK.md updated with progress

## AI-Assisted Development Guidelines

### When Suggesting Code
1. **Context-Aware**: Consider the entire ASI system architecture
2. **Production-Ready**: All code must be production-quality
3. **Performance**: Optimize for scalability (handle 1M+ entities)
4. **Error Handling**: Include comprehensive error handling
5. **Logging**: Add appropriate logging statements

### When Refactoring
1. **Maintain Compatibility**: Don't break existing APIs
2. **Improve Performance**: Look for optimization opportunities
3. **Enhance Readability**: Make code more maintainable
4. **Add Type Hints**: Improve type safety
5. **Update Tests**: Ensure tests still pass

### When Adding Features
1. **Check Existing Code**: Reuse existing components
2. **Follow Patterns**: Use established patterns in the codebase
3. **Update Documentation**: Add to relevant docs
4. **Add Tests**: Include comprehensive tests
5. **Consider Scalability**: Design for growth

## Priority Areas for Development

### 1. Agent Specialization (High Priority)
- Enhance agent capabilities with domain-specific knowledge
- Implement advanced learning algorithms
- Add inter-agent communication protocols
- Optimize task distribution

### 2. Knowledge Graph Enhancement (High Priority)
- Implement graph traversal algorithms
- Add relationship inference
- Optimize query performance
- Implement caching strategies

### 3. Self-Improvement System (Critical)
- Novel algorithm generation
- Code optimization
- Performance monitoring
- Automated testing

### 4. Scalability (High Priority)
- Distributed processing
- Load balancing
- Resource optimization
- Horizontal scaling

### 5. Security (Critical)
- Input validation
- API key management
- Access control
- Audit logging

## Common Patterns

### AWS Integration
```python
from src.integrations.aws_integration import AWSIntegration

aws = AWSIntegration()
await aws.store_entity(entity)
```

### Agent Usage
```python
from src.agents.agent_manager import AgentManager

manager = AgentManager()
await manager.initialize_agents(250)
result = await manager.assign_task(task)
```

### Knowledge Graph
```python
from src.knowledge.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()
await kg.add_entity(entity)
results = await kg.query("search term")
```

## Quality Targets

- **Code Quality**: 100/100
- **Test Coverage**: 95%+
- **Performance**: <100ms query time
- **Success Rate**: 99%+
- **Documentation**: Complete and up-to-date

## Remember

This is a TRUE ASI system aiming for 100% functionality. Every line of code should:
- Be production-ready
- Follow best practices
- Be fully tested
- Be well-documented
- Contribute to the goal of TRUE ASI

**Target**: 100% Fully Functional True Artificial Super Intelligence
