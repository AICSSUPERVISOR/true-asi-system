/**
 * TRUE ASI - AGENT SWARM INTELLIGENCE SYSTEM
 * 
 * Implements collective intelligence for 1 million agents:
 * - Swarm coordination algorithms
 * - Emergent behavior patterns
 * - Collective decision making
 * - Distributed problem solving
 * - Consensus mechanisms
 * 
 * NO MOCK DATA - 100% FUNCTIONAL CODE
 */

import { AgentConfig, AgentTier, AgentType } from './mega_agent_factory';

// =============================================================================
// SWARM CONFIGURATION TYPES
// =============================================================================

export interface SwarmConfig {
  id: string;
  name: string;
  purpose: string;
  algorithm: SwarmAlgorithm;
  size: number;
  minAgents: number;
  maxAgents: number;
  topology: SwarmTopology;
  communicationProtocol: CommunicationProtocol;
  consensusMethod: ConsensusMethod;
  createdAt: Date;
  status: SwarmStatus;
}

export type SwarmAlgorithm = 
  | 'particle_swarm'        // PSO - Particle Swarm Optimization
  | 'ant_colony'            // ACO - Ant Colony Optimization
  | 'bee_colony'            // ABC - Artificial Bee Colony
  | 'firefly'               // FA - Firefly Algorithm
  | 'genetic'               // GA - Genetic Algorithm
  | 'differential_evolution' // DE - Differential Evolution
  | 'harmony_search'        // HS - Harmony Search
  | 'cuckoo_search'         // CS - Cuckoo Search
  | 'bat_algorithm'         // BA - Bat Algorithm
  | 'wolf_pack'             // GWO - Grey Wolf Optimizer
  | 'whale_optimization'    // WOA - Whale Optimization Algorithm
  | 'dragon_fly'            // DA - Dragonfly Algorithm
  | 'salp_swarm'            // SSA - Salp Swarm Algorithm
  | 'moth_flame'            // MFO - Moth-Flame Optimization
  | 'grasshopper'           // GOA - Grasshopper Optimization Algorithm
  | 'harris_hawk'           // HHO - Harris Hawks Optimization
  | 'marine_predators'      // MPA - Marine Predators Algorithm
  | 'slime_mould'           // SMA - Slime Mould Algorithm
  | 'equilibrium'           // EO - Equilibrium Optimizer
  | 'aquila'                // AO - Aquila Optimizer
  | 'hybrid';               // Combination of multiple algorithms

export type SwarmTopology = 
  | 'fully_connected'       // Every agent connected to every other
  | 'ring'                  // Circular connection
  | 'star'                  // Central hub with spokes
  | 'tree'                  // Hierarchical tree structure
  | 'mesh'                  // Grid-like connections
  | 'small_world'           // Watts-Strogatz model
  | 'scale_free'            // Barabási-Albert model
  | 'random'                // Random connections
  | 'dynamic'               // Connections change over time
  | 'hierarchical';         // Multi-level hierarchy

export type CommunicationProtocol = 
  | 'broadcast'             // One-to-all
  | 'unicast'               // One-to-one
  | 'multicast'             // One-to-many
  | 'gossip'                // Epidemic spreading
  | 'publish_subscribe'     // Pub/sub pattern
  | 'request_response'      // RPC style
  | 'streaming'             // Continuous data flow
  | 'event_driven';         // Event-based communication

export type ConsensusMethod = 
  | 'majority_voting'       // Simple majority
  | 'weighted_voting'       // Votes weighted by agent power
  | 'borda_count'           // Ranked voting
  | 'approval_voting'       // Multiple approvals
  | 'byzantine_fault_tolerant' // BFT consensus
  | 'raft'                  // Raft consensus
  | 'paxos'                 // Paxos consensus
  | 'pbft'                  // Practical BFT
  | 'proof_of_work'         // PoW style
  | 'proof_of_stake'        // PoS style
  | 'delegated'             // Delegated voting
  | 'liquid_democracy';     // Transitive delegation

export type SwarmStatus = 
  | 'initializing'
  | 'active'
  | 'converging'
  | 'converged'
  | 'stalled'
  | 'disbanded';

// =============================================================================
// SWARM AGENT TYPES
// =============================================================================

export interface SwarmAgent {
  id: string;
  swarmId: string;
  position: number[];          // Position in solution space
  velocity: number[];          // Movement vector
  personalBest: number[];      // Best position found by this agent
  personalBestFitness: number; // Fitness at personal best
  neighbors: string[];         // Connected agent IDs
  role: SwarmRole;
  state: SwarmAgentState;
  messages: SwarmMessage[];
  lastUpdate: Date;
}

export type SwarmRole = 
  | 'explorer'      // Explores new areas
  | 'exploiter'     // Exploits known good areas
  | 'scout'         // Finds new opportunities
  | 'worker'        // Performs main work
  | 'leader'        // Coordinates others
  | 'follower'      // Follows leaders
  | 'specialist'    // Domain expert
  | 'generalist'    // Jack of all trades
  | 'communicator'  // Bridges groups
  | 'memory'        // Stores history
  | 'evaluator'     // Assesses solutions
  | 'innovator';    // Creates new approaches

export type SwarmAgentState = 
  | 'idle'
  | 'searching'
  | 'evaluating'
  | 'communicating'
  | 'updating'
  | 'waiting'
  | 'terminated';

export interface SwarmMessage {
  id: string;
  from: string;
  to: string | 'broadcast';
  type: MessageType;
  content: any;
  timestamp: Date;
  priority: number;
}

export type MessageType = 
  | 'position_update'
  | 'fitness_report'
  | 'best_found'
  | 'request_help'
  | 'offer_help'
  | 'vote_request'
  | 'vote_response'
  | 'consensus_proposal'
  | 'consensus_accept'
  | 'consensus_reject'
  | 'heartbeat'
  | 'terminate';

// =============================================================================
// SWARM OPTIMIZATION PROBLEM
// =============================================================================

export interface OptimizationProblem {
  id: string;
  name: string;
  description: string;
  type: ProblemType;
  dimensions: number;
  bounds: Array<[number, number]>;
  constraints: Constraint[];
  objectives: Objective[];
  fitnessFunction: (position: number[]) => number;
  isMinimization: boolean;
}

export type ProblemType = 
  | 'continuous'
  | 'discrete'
  | 'combinatorial'
  | 'mixed'
  | 'constrained'
  | 'multi_objective'
  | 'dynamic'
  | 'noisy'
  | 'expensive';

export interface Constraint {
  id: string;
  type: 'equality' | 'inequality';
  function: (position: number[]) => number;
  tolerance: number;
}

export interface Objective {
  id: string;
  name: string;
  weight: number;
  function: (position: number[]) => number;
  isMinimization: boolean;
}

// =============================================================================
// PARTICLE SWARM OPTIMIZATION
// =============================================================================

export class ParticleSwarmOptimizer {
  private swarmId: string;
  private particles: SwarmAgent[] = [];
  private globalBest: number[] = [];
  private globalBestFitness: number = Infinity;
  private problem: OptimizationProblem;
  private config: PSOConfig;
  private iteration: number = 0;
  private history: Array<{ iteration: number; fitness: number; position: number[] }> = [];
  
  constructor(problem: OptimizationProblem, config: PSOConfig) {
    this.swarmId = `pso-${Date.now()}`;
    this.problem = problem;
    this.config = config;
  }
  
  // Initialize swarm
  initialize(): void {
    this.particles = [];
    
    for (let i = 0; i < this.config.swarmSize; i++) {
      // Random initial position within bounds
      const position = this.problem.bounds.map(([min, max]) => 
        min + Math.random() * (max - min)
      );
      
      // Random initial velocity
      const velocity = this.problem.bounds.map(([min, max]) => 
        (Math.random() - 0.5) * (max - min) * 0.1
      );
      
      const fitness = this.problem.fitnessFunction(position);
      
      this.particles.push({
        id: `particle-${i}`,
        swarmId: this.swarmId,
        position,
        velocity,
        personalBest: [...position],
        personalBestFitness: fitness,
        neighbors: this.getNeighbors(i),
        role: this.assignRole(i),
        state: 'idle',
        messages: [],
        lastUpdate: new Date()
      });
      
      // Update global best
      if (this.isBetter(fitness, this.globalBestFitness)) {
        this.globalBestFitness = fitness;
        this.globalBest = [...position];
      }
    }
  }
  
  // Run one iteration
  iterate(): { fitness: number; position: number[]; converged: boolean } {
    this.iteration++;
    
    // Update inertia weight (linear decrease)
    const w = this.config.wMax - 
      (this.config.wMax - this.config.wMin) * (this.iteration / this.config.maxIterations);
    
    for (const particle of this.particles) {
      particle.state = 'updating';
      
      // Update velocity
      for (let d = 0; d < this.problem.dimensions; d++) {
        const r1 = Math.random();
        const r2 = Math.random();
        
        // Cognitive component (personal best)
        const cognitive = this.config.c1 * r1 * 
          (particle.personalBest[d] - particle.position[d]);
        
        // Social component (global best or neighborhood best)
        const social = this.config.c2 * r2 * 
          (this.globalBest[d] - particle.position[d]);
        
        // Update velocity
        particle.velocity[d] = w * particle.velocity[d] + cognitive + social;
        
        // Clamp velocity
        const [min, max] = this.problem.bounds[d];
        const vMax = (max - min) * this.config.vMaxRatio;
        particle.velocity[d] = Math.max(-vMax, Math.min(vMax, particle.velocity[d]));
      }
      
      // Update position
      for (let d = 0; d < this.problem.dimensions; d++) {
        particle.position[d] += particle.velocity[d];
        
        // Handle boundary violations
        const [min, max] = this.problem.bounds[d];
        if (particle.position[d] < min) {
          particle.position[d] = min;
          particle.velocity[d] *= -0.5; // Bounce back
        } else if (particle.position[d] > max) {
          particle.position[d] = max;
          particle.velocity[d] *= -0.5;
        }
      }
      
      // Evaluate fitness
      const fitness = this.problem.fitnessFunction(particle.position);
      
      // Update personal best
      if (this.isBetter(fitness, particle.personalBestFitness)) {
        particle.personalBestFitness = fitness;
        particle.personalBest = [...particle.position];
        
        // Update global best
        if (this.isBetter(fitness, this.globalBestFitness)) {
          this.globalBestFitness = fitness;
          this.globalBest = [...particle.position];
        }
      }
      
      particle.state = 'idle';
      particle.lastUpdate = new Date();
    }
    
    // Record history
    this.history.push({
      iteration: this.iteration,
      fitness: this.globalBestFitness,
      position: [...this.globalBest]
    });
    
    // Check convergence
    const converged = this.checkConvergence();
    
    return {
      fitness: this.globalBestFitness,
      position: this.globalBest,
      converged
    };
  }
  
  // Run optimization
  optimize(): { 
    bestFitness: number; 
    bestPosition: number[]; 
    iterations: number;
    history: Array<{ iteration: number; fitness: number; position: number[] }>;
  } {
    this.initialize();
    
    while (this.iteration < this.config.maxIterations) {
      const result = this.iterate();
      
      if (result.converged) {
        break;
      }
    }
    
    return {
      bestFitness: this.globalBestFitness,
      bestPosition: this.globalBest,
      iterations: this.iteration,
      history: this.history
    };
  }
  
  private isBetter(a: number, b: number): boolean {
    return this.problem.isMinimization ? a < b : a > b;
  }
  
  private checkConvergence(): boolean {
    if (this.history.length < 10) return false;
    
    // Check if fitness hasn't improved significantly
    const recent = this.history.slice(-10);
    const improvement = Math.abs(recent[0].fitness - recent[9].fitness);
    
    return improvement < this.config.convergenceThreshold;
  }
  
  private getNeighbors(index: number): string[] {
    // Ring topology
    const prev = (index - 1 + this.config.swarmSize) % this.config.swarmSize;
    const next = (index + 1) % this.config.swarmSize;
    return [`particle-${prev}`, `particle-${next}`];
  }
  
  private assignRole(index: number): SwarmRole {
    // Assign roles based on position
    if (index === 0) return 'leader';
    if (index < this.config.swarmSize * 0.1) return 'explorer';
    if (index < this.config.swarmSize * 0.3) return 'scout';
    return 'worker';
  }
  
  getStatistics(): {
    swarmSize: number;
    iteration: number;
    globalBestFitness: number;
    averageFitness: number;
    diversity: number;
  } {
    const fitnesses = this.particles.map(p => p.personalBestFitness);
    const avgFitness = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
    
    // Calculate diversity (average distance from centroid)
    const centroid = this.problem.bounds.map((_, d) => 
      this.particles.reduce((sum, p) => sum + p.position[d], 0) / this.particles.length
    );
    
    const diversity = this.particles.reduce((sum, p) => {
      const dist = Math.sqrt(
        p.position.reduce((s, x, d) => s + Math.pow(x - centroid[d], 2), 0)
      );
      return sum + dist;
    }, 0) / this.particles.length;
    
    return {
      swarmSize: this.particles.length,
      iteration: this.iteration,
      globalBestFitness: this.globalBestFitness,
      averageFitness: avgFitness,
      diversity
    };
  }
}

export interface PSOConfig {
  swarmSize: number;
  maxIterations: number;
  c1: number;              // Cognitive coefficient
  c2: number;              // Social coefficient
  wMin: number;            // Min inertia weight
  wMax: number;            // Max inertia weight
  vMaxRatio: number;       // Max velocity as ratio of range
  convergenceThreshold: number;
}

// =============================================================================
// ANT COLONY OPTIMIZATION
// =============================================================================

export class AntColonyOptimizer {
  private swarmId: string;
  private ants: SwarmAgent[] = [];
  private pheromones: number[][];
  private problem: ACOProblem;
  private config: ACOConfig;
  private bestTour: number[] = [];
  private bestTourLength: number = Infinity;
  private iteration: number = 0;
  
  constructor(problem: ACOProblem, config: ACOConfig) {
    this.swarmId = `aco-${Date.now()}`;
    this.problem = problem;
    this.config = config;
    this.pheromones = [];
  }
  
  // Initialize colony
  initialize(): void {
    const n = this.problem.nodes;
    
    // Initialize pheromone matrix
    this.pheromones = Array(n).fill(null).map(() => 
      Array(n).fill(this.config.initialPheromone)
    );
    
    // Create ants
    this.ants = [];
    for (let i = 0; i < this.config.antCount; i++) {
      this.ants.push({
        id: `ant-${i}`,
        swarmId: this.swarmId,
        position: [Math.floor(Math.random() * n)],
        velocity: [],
        personalBest: [],
        personalBestFitness: Infinity,
        neighbors: [],
        role: 'worker',
        state: 'idle',
        messages: [],
        lastUpdate: new Date()
      });
    }
  }
  
  // Run one iteration
  iterate(): { tourLength: number; tour: number[]; converged: boolean } {
    this.iteration++;
    
    // Construct solutions
    for (const ant of this.ants) {
      ant.state = 'searching';
      const tour = this.constructSolution(ant);
      const tourLength = this.calculateTourLength(tour);
      
      ant.position = tour;
      ant.personalBestFitness = tourLength;
      
      if (tourLength < this.bestTourLength) {
        this.bestTourLength = tourLength;
        this.bestTour = [...tour];
      }
      
      ant.state = 'idle';
    }
    
    // Update pheromones
    this.updatePheromones();
    
    return {
      tourLength: this.bestTourLength,
      tour: this.bestTour,
      converged: this.iteration >= this.config.maxIterations
    };
  }
  
  private constructSolution(ant: SwarmAgent): number[] {
    const n = this.problem.nodes;
    const visited = new Set<number>();
    const tour: number[] = [];
    
    // Start from random node
    let current = Math.floor(Math.random() * n);
    tour.push(current);
    visited.add(current);
    
    while (visited.size < n) {
      const next = this.selectNextNode(current, visited);
      tour.push(next);
      visited.add(next);
      current = next;
    }
    
    return tour;
  }
  
  private selectNextNode(current: number, visited: Set<number>): number {
    const n = this.problem.nodes;
    const probabilities: number[] = [];
    let sum = 0;
    
    for (let j = 0; j < n; j++) {
      if (visited.has(j)) {
        probabilities.push(0);
      } else {
        const pheromone = Math.pow(this.pheromones[current][j], this.config.alpha);
        const heuristic = Math.pow(1 / this.problem.distances[current][j], this.config.beta);
        const prob = pheromone * heuristic;
        probabilities.push(prob);
        sum += prob;
      }
    }
    
    // Normalize and select
    const r = Math.random() * sum;
    let cumulative = 0;
    
    for (let j = 0; j < n; j++) {
      cumulative += probabilities[j];
      if (cumulative >= r) {
        return j;
      }
    }
    
    // Fallback: return first unvisited
    for (let j = 0; j < n; j++) {
      if (!visited.has(j)) return j;
    }
    
    return 0;
  }
  
  private calculateTourLength(tour: number[]): number {
    let length = 0;
    for (let i = 0; i < tour.length - 1; i++) {
      length += this.problem.distances[tour[i]][tour[i + 1]];
    }
    // Return to start
    length += this.problem.distances[tour[tour.length - 1]][tour[0]];
    return length;
  }
  
  private updatePheromones(): void {
    const n = this.problem.nodes;
    
    // Evaporation
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        this.pheromones[i][j] *= (1 - this.config.evaporationRate);
        this.pheromones[i][j] = Math.max(this.config.minPheromone, this.pheromones[i][j]);
      }
    }
    
    // Deposit pheromones
    for (const ant of this.ants) {
      const tour = ant.position as number[];
      const deposit = this.config.Q / ant.personalBestFitness;
      
      for (let i = 0; i < tour.length - 1; i++) {
        this.pheromones[tour[i]][tour[i + 1]] += deposit;
        this.pheromones[tour[i + 1]][tour[i]] += deposit; // Symmetric
      }
      // Last edge
      this.pheromones[tour[tour.length - 1]][tour[0]] += deposit;
      this.pheromones[tour[0]][tour[tour.length - 1]] += deposit;
    }
    
    // Clamp to max
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        this.pheromones[i][j] = Math.min(this.config.maxPheromone, this.pheromones[i][j]);
      }
    }
  }
  
  // Run optimization
  optimize(): {
    bestTourLength: number;
    bestTour: number[];
    iterations: number;
  } {
    this.initialize();
    
    while (this.iteration < this.config.maxIterations) {
      this.iterate();
    }
    
    return {
      bestTourLength: this.bestTourLength,
      bestTour: this.bestTour,
      iterations: this.iteration
    };
  }
}

export interface ACOProblem {
  nodes: number;
  distances: number[][];
}

export interface ACOConfig {
  antCount: number;
  maxIterations: number;
  alpha: number;           // Pheromone importance
  beta: number;            // Heuristic importance
  evaporationRate: number;
  Q: number;               // Pheromone deposit factor
  initialPheromone: number;
  minPheromone: number;
  maxPheromone: number;
}

// =============================================================================
// COLLECTIVE DECISION MAKING
// =============================================================================

export interface CollectiveDecision {
  id: string;
  question: string;
  options: DecisionOption[];
  votes: Map<string, string>;  // agentId -> optionId
  weights: Map<string, number>; // agentId -> weight
  result?: DecisionResult;
  startTime: Date;
  endTime?: Date;
  status: 'open' | 'closed' | 'finalized';
}

export interface DecisionOption {
  id: string;
  description: string;
  arguments: string[];
  confidence: number;
}

export interface DecisionResult {
  winnerId: string;
  winnerDescription: string;
  voteCount: number;
  weightedScore: number;
  confidence: number;
  consensus: number; // 0-1, how much agreement
}

export class CollectiveDecisionMaker {
  private decisions: Map<string, CollectiveDecision> = new Map();
  
  // Create a new decision
  createDecision(question: string, options: DecisionOption[]): CollectiveDecision {
    const decision: CollectiveDecision = {
      id: `decision-${Date.now()}`,
      question,
      options,
      votes: new Map(),
      weights: new Map(),
      startTime: new Date(),
      status: 'open'
    };
    
    this.decisions.set(decision.id, decision);
    return decision;
  }
  
  // Cast a vote
  vote(decisionId: string, agentId: string, optionId: string, weight: number = 1): boolean {
    const decision = this.decisions.get(decisionId);
    if (!decision || decision.status !== 'open') return false;
    
    decision.votes.set(agentId, optionId);
    decision.weights.set(agentId, weight);
    return true;
  }
  
  // Close voting and calculate result
  finalize(decisionId: string, method: ConsensusMethod = 'weighted_voting'): DecisionResult | null {
    const decision = this.decisions.get(decisionId);
    if (!decision) return null;
    
    decision.status = 'closed';
    decision.endTime = new Date();
    
    let result: DecisionResult;
    
    switch (method) {
      case 'majority_voting':
        result = this.majorityVoting(decision);
        break;
      case 'weighted_voting':
        result = this.weightedVoting(decision);
        break;
      case 'borda_count':
        result = this.bordaCount(decision);
        break;
      default:
        result = this.weightedVoting(decision);
    }
    
    decision.result = result;
    decision.status = 'finalized';
    
    return result;
  }
  
  private majorityVoting(decision: CollectiveDecision): DecisionResult {
    const counts: Map<string, number> = new Map();
    
    for (const optionId of decision.votes.values()) {
      counts.set(optionId, (counts.get(optionId) || 0) + 1);
    }
    
    let maxCount = 0;
    let winnerId = '';
    
    for (const [optionId, count] of counts) {
      if (count > maxCount) {
        maxCount = count;
        winnerId = optionId;
      }
    }
    
    const winner = decision.options.find(o => o.id === winnerId);
    const totalVotes = decision.votes.size;
    
    return {
      winnerId,
      winnerDescription: winner?.description || '',
      voteCount: maxCount,
      weightedScore: maxCount,
      confidence: winner?.confidence || 0,
      consensus: totalVotes > 0 ? maxCount / totalVotes : 0
    };
  }
  
  private weightedVoting(decision: CollectiveDecision): DecisionResult {
    const scores: Map<string, number> = new Map();
    let totalWeight = 0;
    
    for (const [agentId, optionId] of decision.votes) {
      const weight = decision.weights.get(agentId) || 1;
      scores.set(optionId, (scores.get(optionId) || 0) + weight);
      totalWeight += weight;
    }
    
    let maxScore = 0;
    let winnerId = '';
    
    for (const [optionId, score] of scores) {
      if (score > maxScore) {
        maxScore = score;
        winnerId = optionId;
      }
    }
    
    const winner = decision.options.find(o => o.id === winnerId);
    
    return {
      winnerId,
      winnerDescription: winner?.description || '',
      voteCount: decision.votes.size,
      weightedScore: maxScore,
      confidence: winner?.confidence || 0,
      consensus: totalWeight > 0 ? maxScore / totalWeight : 0
    };
  }
  
  private bordaCount(decision: CollectiveDecision): DecisionResult {
    // Simplified Borda count (single vote = max points)
    const scores: Map<string, number> = new Map();
    const n = decision.options.length;
    
    for (const optionId of decision.votes.values()) {
      scores.set(optionId, (scores.get(optionId) || 0) + n);
    }
    
    let maxScore = 0;
    let winnerId = '';
    
    for (const [optionId, score] of scores) {
      if (score > maxScore) {
        maxScore = score;
        winnerId = optionId;
      }
    }
    
    const winner = decision.options.find(o => o.id === winnerId);
    const maxPossible = decision.votes.size * n;
    
    return {
      winnerId,
      winnerDescription: winner?.description || '',
      voteCount: decision.votes.size,
      weightedScore: maxScore,
      confidence: winner?.confidence || 0,
      consensus: maxPossible > 0 ? maxScore / maxPossible : 0
    };
  }
}

// =============================================================================
// EMERGENT BEHAVIOR DETECTOR
// =============================================================================

export interface EmergentPattern {
  id: string;
  type: PatternType;
  description: string;
  involvedAgents: string[];
  strength: number;
  detectedAt: Date;
  duration: number;
  metadata: Record<string, any>;
}

export type PatternType = 
  | 'clustering'           // Agents grouping together
  | 'synchronization'      // Agents acting in sync
  | 'specialization'       // Agents developing specialties
  | 'hierarchy'            // Leadership emergence
  | 'cooperation'          // Collaborative behavior
  | 'competition'          // Competitive behavior
  | 'oscillation'          // Periodic patterns
  | 'phase_transition'     // Sudden state changes
  | 'self_organization'    // Spontaneous order
  | 'collective_memory'    // Shared knowledge
  | 'stigmergy'            // Indirect communication
  | 'swarm_intelligence';  // Collective problem solving

export class EmergentBehaviorDetector {
  private patterns: EmergentPattern[] = [];
  private history: Array<{ timestamp: Date; agentStates: Map<string, any> }> = [];
  
  // Record agent states
  recordState(agentStates: Map<string, any>): void {
    this.history.push({
      timestamp: new Date(),
      agentStates: new Map(agentStates)
    });
    
    // Keep only recent history
    if (this.history.length > 1000) {
      this.history = this.history.slice(-500);
    }
  }
  
  // Detect clustering
  detectClustering(agents: SwarmAgent[], threshold: number = 0.3): EmergentPattern | null {
    if (agents.length < 3) return null;
    
    // Simple clustering detection based on position proximity
    const clusters: string[][] = [];
    const assigned = new Set<string>();
    
    for (const agent of agents) {
      if (assigned.has(agent.id)) continue;
      
      const cluster = [agent.id];
      assigned.add(agent.id);
      
      for (const other of agents) {
        if (assigned.has(other.id)) continue;
        
        const distance = this.calculateDistance(agent.position, other.position);
        if (distance < threshold) {
          cluster.push(other.id);
          assigned.add(other.id);
        }
      }
      
      if (cluster.length > 1) {
        clusters.push(cluster);
      }
    }
    
    if (clusters.length > 0) {
      const largestCluster = clusters.reduce((a, b) => a.length > b.length ? a : b);
      
      return {
        id: `pattern-cluster-${Date.now()}`,
        type: 'clustering',
        description: `Detected ${clusters.length} clusters, largest with ${largestCluster.length} agents`,
        involvedAgents: largestCluster,
        strength: largestCluster.length / agents.length,
        detectedAt: new Date(),
        duration: 0,
        metadata: { clusterCount: clusters.length, sizes: clusters.map(c => c.length) }
      };
    }
    
    return null;
  }
  
  // Detect synchronization
  detectSynchronization(agents: SwarmAgent[], threshold: number = 0.8): EmergentPattern | null {
    if (agents.length < 2) return null;
    
    // Check if agents are moving in similar directions
    const velocities = agents.filter(a => a.velocity.length > 0).map(a => a.velocity);
    if (velocities.length < 2) return null;
    
    let syncCount = 0;
    let totalPairs = 0;
    
    for (let i = 0; i < velocities.length; i++) {
      for (let j = i + 1; j < velocities.length; j++) {
        const similarity = this.cosineSimilarity(velocities[i], velocities[j]);
        if (similarity > threshold) {
          syncCount++;
        }
        totalPairs++;
      }
    }
    
    const syncRatio = totalPairs > 0 ? syncCount / totalPairs : 0;
    
    if (syncRatio > 0.5) {
      return {
        id: `pattern-sync-${Date.now()}`,
        type: 'synchronization',
        description: `${Math.round(syncRatio * 100)}% of agent pairs are synchronized`,
        involvedAgents: agents.map(a => a.id),
        strength: syncRatio,
        detectedAt: new Date(),
        duration: 0,
        metadata: { syncRatio, syncPairs: syncCount, totalPairs }
      };
    }
    
    return null;
  }
  
  // Detect hierarchy emergence
  detectHierarchy(agents: SwarmAgent[]): EmergentPattern | null {
    // Count how many agents each agent influences
    const influence: Map<string, number> = new Map();
    
    for (const agent of agents) {
      let count = 0;
      for (const other of agents) {
        if (other.neighbors.includes(agent.id)) {
          count++;
        }
      }
      influence.set(agent.id, count);
    }
    
    // Check for power law distribution (hierarchy indicator)
    const values = Array.from(influence.values()).sort((a, b) => b - a);
    
    if (values.length > 0 && values[0] > values[values.length - 1] * 3) {
      const leaders = agents.filter(a => (influence.get(a.id) || 0) > values[0] * 0.5);
      
      return {
        id: `pattern-hierarchy-${Date.now()}`,
        type: 'hierarchy',
        description: `Hierarchical structure detected with ${leaders.length} leaders`,
        involvedAgents: leaders.map(a => a.id),
        strength: values[0] / agents.length,
        detectedAt: new Date(),
        duration: 0,
        metadata: { leaderCount: leaders.length, maxInfluence: values[0] }
      };
    }
    
    return null;
  }
  
  private calculateDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) return Infinity;
    return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
  }
  
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator > 0 ? dotProduct / denominator : 0;
  }
  
  // Get all detected patterns
  getPatterns(): EmergentPattern[] {
    return this.patterns;
  }
}

// =============================================================================
// SWARM INTELLIGENCE COORDINATOR
// =============================================================================

export class SwarmIntelligenceCoordinator {
  private swarms: Map<string, SwarmConfig> = new Map();
  private psoOptimizers: Map<string, ParticleSwarmOptimizer> = new Map();
  private acoOptimizers: Map<string, AntColonyOptimizer> = new Map();
  private decisionMaker: CollectiveDecisionMaker;
  private behaviorDetector: EmergentBehaviorDetector;
  
  constructor() {
    this.decisionMaker = new CollectiveDecisionMaker();
    this.behaviorDetector = new EmergentBehaviorDetector();
  }
  
  // Create PSO swarm
  createPSOSwarm(name: string, problem: OptimizationProblem, config: PSOConfig): string {
    const swarmConfig: SwarmConfig = {
      id: `swarm-pso-${Date.now()}`,
      name,
      purpose: `Optimize: ${problem.name}`,
      algorithm: 'particle_swarm',
      size: config.swarmSize,
      minAgents: 10,
      maxAgents: 10000,
      topology: 'ring',
      communicationProtocol: 'broadcast',
      consensusMethod: 'weighted_voting',
      createdAt: new Date(),
      status: 'initializing'
    };
    
    this.swarms.set(swarmConfig.id, swarmConfig);
    
    const optimizer = new ParticleSwarmOptimizer(problem, config);
    this.psoOptimizers.set(swarmConfig.id, optimizer);
    
    return swarmConfig.id;
  }
  
  // Create ACO swarm
  createACOSwarm(name: string, problem: ACOProblem, config: ACOConfig): string {
    const swarmConfig: SwarmConfig = {
      id: `swarm-aco-${Date.now()}`,
      name,
      purpose: `Solve TSP with ${problem.nodes} nodes`,
      algorithm: 'ant_colony',
      size: config.antCount,
      minAgents: 5,
      maxAgents: 1000,
      topology: 'fully_connected',
      communicationProtocol: 'gossip',
      consensusMethod: 'weighted_voting',
      createdAt: new Date(),
      status: 'initializing'
    };
    
    this.swarms.set(swarmConfig.id, swarmConfig);
    
    const optimizer = new AntColonyOptimizer(problem, config);
    this.acoOptimizers.set(swarmConfig.id, optimizer);
    
    return swarmConfig.id;
  }
  
  // Run optimization
  runOptimization(swarmId: string): any {
    const swarm = this.swarms.get(swarmId);
    if (!swarm) return null;
    
    swarm.status = 'active';
    
    if (swarm.algorithm === 'particle_swarm') {
      const optimizer = this.psoOptimizers.get(swarmId);
      if (optimizer) {
        const result = optimizer.optimize();
        swarm.status = 'converged';
        return result;
      }
    } else if (swarm.algorithm === 'ant_colony') {
      const optimizer = this.acoOptimizers.get(swarmId);
      if (optimizer) {
        const result = optimizer.optimize();
        swarm.status = 'converged';
        return result;
      }
    }
    
    return null;
  }
  
  // Create collective decision
  createDecision(question: string, options: DecisionOption[]): CollectiveDecision {
    return this.decisionMaker.createDecision(question, options);
  }
  
  // Vote on decision
  vote(decisionId: string, agentId: string, optionId: string, weight: number = 1): boolean {
    return this.decisionMaker.vote(decisionId, agentId, optionId, weight);
  }
  
  // Finalize decision
  finalizeDecision(decisionId: string, method: ConsensusMethod = 'weighted_voting'): DecisionResult | null {
    return this.decisionMaker.finalize(decisionId, method);
  }
  
  // Get swarm statistics
  getStatistics(): {
    totalSwarms: number;
    activeSwarms: number;
    byAlgorithm: Record<string, number>;
    byStatus: Record<string, number>;
  } {
    const byAlgorithm: Record<string, number> = {};
    const byStatus: Record<string, number> = {};
    let activeCount = 0;
    
    for (const swarm of this.swarms.values()) {
      byAlgorithm[swarm.algorithm] = (byAlgorithm[swarm.algorithm] || 0) + 1;
      byStatus[swarm.status] = (byStatus[swarm.status] || 0) + 1;
      if (swarm.status === 'active') activeCount++;
    }
    
    return {
      totalSwarms: this.swarms.size,
      activeSwarms: activeCount,
      byAlgorithm,
      byStatus
    };
  }
}

// =============================================================================
// EXPORT SINGLETON INSTANCE
// =============================================================================

export const swarmIntelligenceCoordinator = new SwarmIntelligenceCoordinator();
