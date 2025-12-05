/**
 * Unit Tests for AI Model Router
 * 
 * Tests model selection, ranking, and task type matching
 */

import { describe, it, expect } from 'vitest';
import {
  selectModelsForTask,
  getModelById,
  getModelsByProvider,
  AI_MODEL_REGISTRY,
  type AIModel,
  type TaskType,
} from './helpers/ai_model_router';

describe('AI Model Router', () => {
  describe('Model Selection', () => {
    it('should select best model for strategy task', () => {
      const models = selectModelsForTask('strategy', 1);
      const model = models[0];
      
      expect(model).toBeDefined();
      expect(model.capabilities).toContain('strategy');
      expect(model.weight).toBeGreaterThan(90); // Should be high-tier model
    });

    it('should select best model for writing task', () => {
      const models = selectModelsForTask('writing', 1);
      const model = models[0];
      
      expect(model).toBeDefined();
      expect(model.capabilities).toContain('writing');
    });

    it('should select best model for analysis task', () => {
      const models = selectModelsForTask('analysis', 1);
      const model = models[0];
      
      expect(model).toBeDefined();
      expect(model.capabilities).toContain('analysis');
    });

    it('should select best model for coding task', () => {
      const models = selectModelsForTask('coding', 1);
      const model = models[0];
      
      expect(model).toBeDefined();
      expect(model.capabilities).toContain('coding');
    });

    it('should select superintelligence model when available', () => {
      const models = selectModelsForTask('superintelligence', 1);
      const model = models[0];
      
      expect(model).toBeDefined();
      expect(model.capabilities).toContain('superintelligence');
      expect(model.weight).toBe(100); // Highest weight
    });
  });

  describe('Model Filtering by Task Type', () => {
    it('should return all strategy models', () => {
      const models = selectModelsForTask('strategy', 100);
      
      expect(models.length).toBeGreaterThan(0);
      models.forEach(model => {
        expect(model.capabilities).toContain('strategy');
      });
    });

    it('should return all coding models', () => {
      const models = selectModelsForTask('coding', 100);
      
      expect(models.length).toBeGreaterThan(0);
      models.forEach(model => {
        expect(model.capabilities).toContain('coding');
      });
    });

    it('should return all financial models', () => {
      const models = selectModelsForTask('financial', 100);
      
      expect(models.length).toBeGreaterThan(0);
      models.forEach(model => {
        expect(model.capabilities).toContain('financial');
      });
    });

    it('should return models sorted by weight descending', () => {
      const models = selectModelsForTask('strategy', 100);
      
      for (let i = 0; i < models.length - 1; i++) {
        expect(models[i].weight).toBeGreaterThanOrEqual(models[i + 1].weight);
      }
    });
  });

  describe('Model Lookup by ID', () => {
    it('should find ASI1 Ultra model', () => {
      const model = getModelById('asi1-ultra');
      
      expect(model).toBeDefined();
      expect(model?.id).toBe('asi1-ultra');
      expect(model?.name).toBe('ASI1 Ultra');
    });

    it('should return undefined for non-existent model', () => {
      const model = getModelById('non-existent-model-xyz');
      
      expect(model).toBeUndefined();
    });
  });

  describe('Model Registry', () => {
    it('should have at least 10 models', () => {
      const models = AI_MODEL_REGISTRY;
      
      expect(models.length).toBeGreaterThanOrEqual(10);
    });

    it('should have unique model IDs', () => {
      const models = AI_MODEL_REGISTRY;
      const ids = models.map(m => m.id);
      const uniqueIds = new Set(ids);
      
      expect(uniqueIds.size).toBe(ids.length);
    });

    it('should have valid weight ranges (0-100)', () => {
      const models = AI_MODEL_REGISTRY;
      
      models.forEach(model => {
        expect(model.weight).toBeGreaterThanOrEqual(0);
        expect(model.weight).toBeLessThanOrEqual(100);
      });
    });

    it('should have valid cost ranges', () => {
      const models = AI_MODEL_REGISTRY;
      
      models.forEach(model => {
        expect(model.costPerToken).toBeGreaterThanOrEqual(0);
        expect(model.costPerToken).toBeLessThan(1); // Less than $1 per token
      });
    });

    it('should have valid speed classifications', () => {
      const models = AI_MODEL_REGISTRY;
      const validSpeeds = ['fast', 'medium', 'slow'];
      
      models.forEach(model => {
        expect(validSpeeds).toContain(model.speed);
      });
    });

    it('should have valid max token ranges', () => {
      const models = AI_MODEL_REGISTRY;
      
      models.forEach(model => {
        expect(model.maxTokens).toBeGreaterThan(0);
        expect(model.maxTokens).toBeLessThanOrEqual(2000000); // 2M max
      });
    });
  });

  describe('Task Type Coverage', () => {
    it('should have models for all task types', () => {
      const taskTypes: TaskType[] = [
        'strategy',
        'writing',
        'analysis',
        'coding',
        'superintelligence',
        'financial',
        'marketing',
        'operations',
        'legal',
        'technical',
      ];

      taskTypes.forEach(taskType => {
        const models = selectModelsForTask(taskType, 100);
        expect(models.length).toBeGreaterThan(0);
      });
    });

    it('should have at least 2 models per major task type', () => {
      const majorTaskTypes: TaskType[] = ['strategy', 'writing', 'analysis', 'coding'];

      majorTaskTypes.forEach(taskType => {
        const models = selectModelsForTask(taskType, 100);
        expect(models.length).toBeGreaterThanOrEqual(2);
      });
    });
  });

  describe('Model Properties', () => {
    it('should have required properties for all models', () => {
      const models = AI_MODEL_REGISTRY;
      
      models.forEach(model => {
        expect(model.id).toBeDefined();
        expect(model.name).toBeDefined();
        expect(model.provider).toBeDefined();
        expect(model.weight).toBeDefined();
        expect(model.costPerToken).toBeDefined();
        expect(model.speed).toBeDefined();
        expect(model.maxTokens).toBeDefined();
        expect(model.capabilities).toBeDefined();
        expect(Array.isArray(model.capabilities)).toBe(true);
      });
    });

    it('should have valid provider names', () => {
      const models = AI_MODEL_REGISTRY;
      const validProviders = ['aiml', 'asi1'];

      models.forEach(model => {
        expect(validProviders).toContain(model.provider);
      });
    });
  });

  describe('Provider Filtering', () => {
    it('should filter models by AIML provider', () => {
      const models = getModelsByProvider('aiml');
      
      expect(models.length).toBeGreaterThan(0);
      models.forEach(model => {
        expect(model.provider).toBe('aiml');
      });
    });

    it('should filter models by ASI1 provider', () => {
      const models = getModelsByProvider('asi1');
      
      expect(models.length).toBeGreaterThan(0);
      models.forEach(model => {
        expect(model.provider).toBe('asi1');
      });
    });
  });

  describe('Performance Characteristics', () => {
    it('should have fast models available', () => {
      const fastModels = AI_MODEL_REGISTRY.filter(m => m.speed === 'fast');
      
      expect(fastModels.length).toBeGreaterThan(0);
    });

    it('should have low-cost models available', () => {
      const lowCostModels = AI_MODEL_REGISTRY.filter(m => m.costPerToken < 0.001);
      
      expect(lowCostModels.length).toBeGreaterThan(0);
    });

    it('should have high-capacity models (>100k tokens)', () => {
      const highCapacityModels = AI_MODEL_REGISTRY.filter(m => m.maxTokens > 100000);
      
      expect(highCapacityModels.length).toBeGreaterThan(0);
    });
  });
});
