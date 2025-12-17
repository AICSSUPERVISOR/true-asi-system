# TRUE ASI System - API Reference

## 1. ASI Engine API

### `ASIEngine.process_task(task: Dict) -> Dict`

Processes a task using the full capabilities of the ASI system.

- **task**: A dictionary containing task details (`type`, `query`, `context`).
- **Returns**: A dictionary with the task result.

## 2. Agent Manager API

### `AgentManager.assign_task(task: Dict) -> Dict`

Assigns a task to an available agent.

- **task**: The task to be assigned.
- **Returns**: The result from the agent or a `queued` status.

## 3. Knowledge Graph API

### `KnowledgeGraph.add_entity(entity: Dict)`

Adds an entity to the knowledge graph.

- **entity**: The entity to add.

### `KnowledgeGraph.query(query: str) -> List[Dict]`

Queries the knowledge graph.

- **query**: The search query.
- **Returns**: A list of matching entities.

