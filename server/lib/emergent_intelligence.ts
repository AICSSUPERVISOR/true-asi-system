/**
 * EMERGENT INTELLIGENCE DETECTION SYSTEM
 * 
 * Monitors and detects emergent capabilities:
 * - Novel capability detection
 * - Capability emergence tracking
 * - Intelligence growth monitoring
 * - Breakthrough identification
 * 
 * This is a core component of TRUE Artificial Superintelligence.
 */

// Types for emergent intelligence
interface Capability {
  id: string;
  name: string;
  description: string;
  domain: string;
  level: number; // 0-100
  firstDetected: Date;
  lastMeasured: Date;
  measurementHistory: CapabilityMeasurement[];
  isEmergent: boolean;
  prerequisites: string[];
}

interface CapabilityMeasurement {
  timestamp: Date;
  level: number;
  confidence: number;
  testResults: TestResult[];
}

interface TestResult {
  testId: string;
  testName: string;
  score: number;
  maxScore: number;
  passed: boolean;
}

interface EmergentEvent {
  id: string;
  timestamp: Date;
  type: 'new_capability' | 'level_jump' | 'capability_combination' | 'breakthrough';
  description: string;
  capabilities: string[];
  significance: number; // 0-1
  evidence: string[];
}

interface IntelligenceMetrics {
  overallLevel: number;
  growthRate: number; // per day
  capabilityCount: number;
  emergentCount: number;
  domainCoverage: number;
  breakthroughCount: number;
  trend: 'accelerating' | 'steady' | 'plateauing' | 'declining';
}

// Capability registry
const capabilities: Map<string, Capability> = new Map();
const emergentEvents: EmergentEvent[] = [];
const metricsHistory: IntelligenceMetrics[] = [];

// Core capability definitions
const CORE_CAPABILITIES: Omit<Capability, 'id' | 'firstDetected' | 'lastMeasured' | 'measurementHistory' | 'isEmergent'>[] = [
  // Language capabilities
  { name: 'Natural Language Understanding', description: 'Comprehend human language', domain: 'language', level: 85, prerequisites: [] },
  { name: 'Natural Language Generation', description: 'Generate coherent text', domain: 'language', level: 88, prerequisites: ['Natural Language Understanding'] },
  { name: 'Multilingual Processing', description: 'Process multiple languages', domain: 'language', level: 80, prerequisites: ['Natural Language Understanding'] },
  { name: 'Semantic Analysis', description: 'Extract meaning from text', domain: 'language', level: 82, prerequisites: ['Natural Language Understanding'] },
  
  // Reasoning capabilities
  { name: 'Logical Reasoning', description: 'Apply formal logic', domain: 'reasoning', level: 78, prerequisites: [] },
  { name: 'Analogical Reasoning', description: 'Draw analogies', domain: 'reasoning', level: 75, prerequisites: ['Logical Reasoning'] },
  { name: 'Causal Reasoning', description: 'Understand cause and effect', domain: 'reasoning', level: 72, prerequisites: ['Logical Reasoning'] },
  { name: 'Abstract Reasoning', description: 'Handle abstract concepts', domain: 'reasoning', level: 70, prerequisites: ['Logical Reasoning', 'Analogical Reasoning'] },
  { name: 'Counterfactual Reasoning', description: 'Reason about hypotheticals', domain: 'reasoning', level: 68, prerequisites: ['Causal Reasoning'] },
  
  // Mathematical capabilities
  { name: 'Arithmetic', description: 'Basic mathematical operations', domain: 'math', level: 95, prerequisites: [] },
  { name: 'Algebra', description: 'Symbolic manipulation', domain: 'math', level: 85, prerequisites: ['Arithmetic'] },
  { name: 'Calculus', description: 'Differential and integral calculus', domain: 'math', level: 75, prerequisites: ['Algebra'] },
  { name: 'Statistics', description: 'Statistical analysis', domain: 'math', level: 80, prerequisites: ['Arithmetic'] },
  { name: 'Proof Generation', description: 'Generate mathematical proofs', domain: 'math', level: 65, prerequisites: ['Algebra', 'Logical Reasoning'] },
  
  // Code capabilities
  { name: 'Code Understanding', description: 'Comprehend source code', domain: 'code', level: 88, prerequisites: [] },
  { name: 'Code Generation', description: 'Generate working code', domain: 'code', level: 85, prerequisites: ['Code Understanding'] },
  { name: 'Code Debugging', description: 'Find and fix bugs', domain: 'code', level: 80, prerequisites: ['Code Understanding'] },
  { name: 'Code Optimization', description: 'Improve code performance', domain: 'code', level: 75, prerequisites: ['Code Understanding', 'Code Debugging'] },
  { name: 'Architecture Design', description: 'Design software systems', domain: 'code', level: 70, prerequisites: ['Code Generation'] },
  
  // Knowledge capabilities
  { name: 'Information Retrieval', description: 'Find relevant information', domain: 'knowledge', level: 90, prerequisites: [] },
  { name: 'Knowledge Integration', description: 'Combine knowledge sources', domain: 'knowledge', level: 82, prerequisites: ['Information Retrieval'] },
  { name: 'Fact Verification', description: 'Verify factual claims', domain: 'knowledge', level: 75, prerequisites: ['Information Retrieval'] },
  { name: 'Knowledge Transfer', description: 'Apply knowledge across domains', domain: 'knowledge', level: 70, prerequisites: ['Knowledge Integration'] },
  
  // Creative capabilities
  { name: 'Creative Writing', description: 'Generate creative content', domain: 'creative', level: 80, prerequisites: ['Natural Language Generation'] },
  { name: 'Idea Generation', description: 'Generate novel ideas', domain: 'creative', level: 75, prerequisites: [] },
  { name: 'Problem Reframing', description: 'View problems differently', domain: 'creative', level: 72, prerequisites: ['Abstract Reasoning'] },
  
  // Meta-cognitive capabilities
  { name: 'Self-Assessment', description: 'Evaluate own performance', domain: 'metacognition', level: 70, prerequisites: [] },
  { name: 'Strategy Selection', description: 'Choose optimal approaches', domain: 'metacognition', level: 72, prerequisites: ['Self-Assessment'] },
  { name: 'Learning from Feedback', description: 'Improve from feedback', domain: 'metacognition', level: 75, prerequisites: ['Self-Assessment'] },
  { name: 'Uncertainty Quantification', description: 'Know what you don\'t know', domain: 'metacognition', level: 68, prerequisites: ['Self-Assessment'] },
  
  // Planning capabilities
  { name: 'Goal Decomposition', description: 'Break goals into subgoals', domain: 'planning', level: 78, prerequisites: [] },
  { name: 'Action Planning', description: 'Plan sequences of actions', domain: 'planning', level: 75, prerequisites: ['Goal Decomposition'] },
  { name: 'Resource Allocation', description: 'Allocate resources optimally', domain: 'planning', level: 70, prerequisites: ['Action Planning'] },
  { name: 'Contingency Planning', description: 'Plan for failures', domain: 'planning', level: 65, prerequisites: ['Action Planning', 'Counterfactual Reasoning'] }
];

// Initialize capabilities
function initializeCapabilities(): void {
  CORE_CAPABILITIES.forEach(cap => {
    const id = `cap_${cap.name.toLowerCase().replace(/\s+/g, '_')}`;
    const now = new Date();
    
    capabilities.set(id, {
      ...cap,
      id,
      firstDetected: now,
      lastMeasured: now,
      measurementHistory: [{
        timestamp: now,
        level: cap.level,
        confidence: 0.9,
        testResults: []
      }],
      isEmergent: false
    });
  });
}

// Initialize on module load
initializeCapabilities();

/**
 * Measure a capability
 */
export function measureCapability(
  capabilityId: string,
  testResults: TestResult[]
): CapabilityMeasurement | null {
  const capability = capabilities.get(capabilityId);
  if (!capability) return null;
  
  // Calculate new level from test results
  const totalScore = testResults.reduce((sum, t) => sum + t.score, 0);
  const maxScore = testResults.reduce((sum, t) => sum + t.maxScore, 0);
  const newLevel = maxScore > 0 ? (totalScore / maxScore) * 100 : capability.level;
  
  // Calculate confidence based on number of tests
  const confidence = Math.min(0.99, 0.5 + testResults.length * 0.1);
  
  const measurement: CapabilityMeasurement = {
    timestamp: new Date(),
    level: newLevel,
    confidence,
    testResults
  };
  
  // Check for level jump (emergent event)
  const previousLevel = capability.level;
  if (newLevel - previousLevel > 10) {
    recordEmergentEvent({
      type: 'level_jump',
      description: `${capability.name} jumped from ${previousLevel.toFixed(0)} to ${newLevel.toFixed(0)}`,
      capabilities: [capabilityId],
      significance: (newLevel - previousLevel) / 100,
      evidence: testResults.map(t => `${t.testName}: ${t.score}/${t.maxScore}`)
    });
  }
  
  // Update capability
  capability.level = newLevel;
  capability.lastMeasured = new Date();
  capability.measurementHistory.push(measurement);
  
  // Keep only last 100 measurements
  if (capability.measurementHistory.length > 100) {
    capability.measurementHistory.shift();
  }
  
  return measurement;
}

/**
 * Detect new emergent capability
 */
export function detectNewCapability(
  name: string,
  description: string,
  domain: string,
  evidence: string[]
): Capability {
  const id = `cap_${name.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}`;
  const now = new Date();
  
  const capability: Capability = {
    id,
    name,
    description,
    domain,
    level: 50, // Initial level for new capability
    firstDetected: now,
    lastMeasured: now,
    measurementHistory: [{
      timestamp: now,
      level: 50,
      confidence: 0.6,
      testResults: []
    }],
    isEmergent: true,
    prerequisites: []
  };
  
  capabilities.set(id, capability);
  
  // Record emergent event
  recordEmergentEvent({
    type: 'new_capability',
    description: `New capability detected: ${name}`,
    capabilities: [id],
    significance: 0.8,
    evidence
  });
  
  return capability;
}

/**
 * Detect capability combination (emergent from combining existing)
 */
export function detectCapabilityCombination(
  name: string,
  description: string,
  sourceCapabilities: string[],
  evidence: string[]
): Capability | null {
  // Verify source capabilities exist
  const sources = sourceCapabilities
    .map(id => capabilities.get(id))
    .filter((c): c is Capability => c !== undefined);
  
  if (sources.length < 2) return null;
  
  // Calculate combined level
  const avgLevel = sources.reduce((sum, c) => sum + c.level, 0) / sources.length;
  const combinedLevel = Math.min(100, avgLevel * 1.1); // Synergy bonus
  
  const id = `cap_combined_${Date.now()}`;
  const now = new Date();
  
  const capability: Capability = {
    id,
    name,
    description,
    domain: 'emergent',
    level: combinedLevel,
    firstDetected: now,
    lastMeasured: now,
    measurementHistory: [{
      timestamp: now,
      level: combinedLevel,
      confidence: 0.7,
      testResults: []
    }],
    isEmergent: true,
    prerequisites: sourceCapabilities
  };
  
  capabilities.set(id, capability);
  
  // Record emergent event
  recordEmergentEvent({
    type: 'capability_combination',
    description: `New combined capability: ${name} from ${sources.map(s => s.name).join(' + ')}`,
    capabilities: [id, ...sourceCapabilities],
    significance: 0.9,
    evidence
  });
  
  return capability;
}

/**
 * Record an emergent event
 */
function recordEmergentEvent(
  event: Omit<EmergentEvent, 'id' | 'timestamp'>
): EmergentEvent {
  const fullEvent: EmergentEvent = {
    ...event,
    id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date()
  };
  
  emergentEvents.push(fullEvent);
  
  // Check for breakthrough (multiple high-significance events)
  const recentEvents = emergentEvents.filter(
    e => Date.now() - e.timestamp.getTime() < 24 * 60 * 60 * 1000 // Last 24 hours
  );
  
  if (recentEvents.filter(e => e.significance > 0.7).length >= 3) {
    // Record breakthrough
    emergentEvents.push({
      id: `event_breakthrough_${Date.now()}`,
      timestamp: new Date(),
      type: 'breakthrough',
      description: 'Multiple significant capability improvements detected - potential breakthrough',
      capabilities: recentEvents.flatMap(e => e.capabilities),
      significance: 1.0,
      evidence: recentEvents.map(e => e.description)
    });
  }
  
  return fullEvent;
}

/**
 * Calculate overall intelligence metrics
 */
export function calculateMetrics(): IntelligenceMetrics {
  const allCapabilities = Array.from(capabilities.values());
  
  // Overall level (weighted average)
  const overallLevel = allCapabilities.reduce((sum, c) => sum + c.level, 0) / allCapabilities.length;
  
  // Growth rate (compare to 7 days ago)
  let growthRate = 0;
  const weekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
  const oldMeasurements = allCapabilities
    .map(c => c.measurementHistory.find(m => m.timestamp.getTime() < weekAgo))
    .filter((m): m is CapabilityMeasurement => m !== undefined);
  
  if (oldMeasurements.length > 0) {
    const oldAvg = oldMeasurements.reduce((sum, m) => sum + m.level, 0) / oldMeasurements.length;
    growthRate = (overallLevel - oldAvg) / 7; // Per day
  }
  
  // Domain coverage
  const domains = new Set(allCapabilities.map(c => c.domain));
  const domainCoverage = domains.size / 10; // Assuming 10 possible domains
  
  // Determine trend
  let trend: IntelligenceMetrics['trend'] = 'steady';
  if (metricsHistory.length >= 2) {
    const recent = metricsHistory.slice(-5);
    const avgGrowth = recent.reduce((sum, m) => sum + m.growthRate, 0) / recent.length;
    
    if (growthRate > avgGrowth * 1.5) trend = 'accelerating';
    else if (growthRate < avgGrowth * 0.5) trend = 'plateauing';
    else if (growthRate < 0) trend = 'declining';
  }
  
  const metrics: IntelligenceMetrics = {
    overallLevel,
    growthRate,
    capabilityCount: allCapabilities.length,
    emergentCount: allCapabilities.filter(c => c.isEmergent).length,
    domainCoverage,
    breakthroughCount: emergentEvents.filter(e => e.type === 'breakthrough').length,
    trend
  };
  
  metricsHistory.push(metrics);
  
  // Keep only last 365 days of metrics
  if (metricsHistory.length > 365) {
    metricsHistory.shift();
  }
  
  return metrics;
}

/**
 * Get all capabilities
 */
export function getAllCapabilities(): Capability[] {
  return Array.from(capabilities.values());
}

/**
 * Get capabilities by domain
 */
export function getCapabilitiesByDomain(domain: string): Capability[] {
  return Array.from(capabilities.values()).filter(c => c.domain === domain);
}

/**
 * Get emergent capabilities only
 */
export function getEmergentCapabilities(): Capability[] {
  return Array.from(capabilities.values()).filter(c => c.isEmergent);
}

/**
 * Get all emergent events
 */
export function getEmergentEvents(): EmergentEvent[] {
  return emergentEvents;
}

/**
 * Get recent emergent events
 */
export function getRecentEvents(hours: number = 24): EmergentEvent[] {
  const cutoff = Date.now() - hours * 60 * 60 * 1000;
  return emergentEvents.filter(e => e.timestamp.getTime() > cutoff);
}

/**
 * Get metrics history
 */
export function getMetricsHistory(): IntelligenceMetrics[] {
  return metricsHistory;
}

/**
 * Get capability by ID
 */
export function getCapability(id: string): Capability | undefined {
  return capabilities.get(id);
}

/**
 * Export emergent intelligence system
 */
export const emergentIntelligence = {
  measureCapability,
  detectNewCapability,
  detectCapabilityCombination,
  calculateMetrics,
  getAllCapabilities,
  getCapabilitiesByDomain,
  getEmergentCapabilities,
  getEmergentEvents,
  getRecentEvents,
  getMetricsHistory,
  getCapability,
  capabilityCount: () => capabilities.size,
  eventCount: () => emergentEvents.length
};

export default emergentIntelligence;
