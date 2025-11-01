# TRUE ASI System - Knowledge Graph

## 1. Overview

The Knowledge Hypergraph is a dynamic, multi-dimensional knowledge base that stores over 61,792 entities and their relationships.

## 2. Schema

- **Entities**: Stored in the `asi-knowledge-graph-entities` DynamoDB table.
- **Relationships**: Stored in the `asi-knowledge-graph-relationships` DynamoDB table.

## 3. Operations

- **`add_entity`**: Adds a new entity.
- **`add_relationship`**: Creates a link between two entities.
- **`query`**: Searches for entities based on a query.

