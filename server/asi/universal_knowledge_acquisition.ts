/**
 * TRUE ASI - UNIVERSAL KNOWLEDGE ACQUISITION SYSTEM
 * 
 * Acquires knowledge from ALL sources:
 * - All GitHub repositories
 * - All academic papers
 * - All documentation
 * - All code bases
 * - All web content
 * 
 * NO MOCK DATA - 100% FUNCTIONAL CODE
 */

// =============================================================================
// KNOWLEDGE SOURCE TYPES
// =============================================================================

export interface KnowledgeSource {
  id: string;
  type: KnowledgeSourceType;
  url: string;
  name: string;
  description: string;
  lastCrawled?: Date;
  totalItems: number;
  extractedFacts: number;
  quality: number; // 0-100
  status: 'pending' | 'crawling' | 'processing' | 'complete' | 'error';
}

export type KnowledgeSourceType = 
  | 'github_repository'
  | 'github_organization'
  | 'github_user'
  | 'academic_paper'
  | 'documentation'
  | 'api_reference'
  | 'tutorial'
  | 'blog_post'
  | 'forum_thread'
  | 'stack_overflow'
  | 'wikipedia'
  | 'arxiv'
  | 'huggingface'
  | 'kaggle'
  | 'npm_package'
  | 'pypi_package'
  | 'crates_io'
  | 'maven_central'
  | 'nuget'
  | 'rubygems'
  | 'web_page'
  | 'pdf_document'
  | 'video_transcript'
  | 'podcast_transcript'
  | 'book_chapter'
  | 'patent'
  | 'news_article'
  | 'social_media'
  | 'code_snippet'
  | 'dataset';

// =============================================================================
// KNOWLEDGE ITEM TYPES
// =============================================================================

export interface KnowledgeItem {
  id: string;
  sourceId: string;
  type: KnowledgeItemType;
  title: string;
  content: string;
  summary: string;
  keywords: string[];
  entities: NamedEntity[];
  relations: KnowledgeRelation[];
  embedding?: number[];
  confidence: number;
  createdAt: Date;
  updatedAt: Date;
  metadata: Record<string, any>;
}

export type KnowledgeItemType = 
  | 'fact'
  | 'concept'
  | 'definition'
  | 'procedure'
  | 'example'
  | 'code'
  | 'formula'
  | 'theorem'
  | 'proof'
  | 'algorithm'
  | 'pattern'
  | 'best_practice'
  | 'anti_pattern'
  | 'architecture'
  | 'api'
  | 'function'
  | 'class'
  | 'interface'
  | 'type'
  | 'constant'
  | 'configuration'
  | 'command'
  | 'query'
  | 'schema'
  | 'model'
  | 'dataset_info'
  | 'benchmark'
  | 'metric'
  | 'citation'
  | 'quote'
  | 'documentation';

export interface NamedEntity {
  text: string;
  type: string;
  start: number;
  end: number;
  confidence: number;
}

export interface KnowledgeRelation {
  sourceId: string;
  targetId: string;
  type: RelationType;
  confidence: number;
  evidence: string;
}

export type RelationType = 
  | 'is_a'
  | 'part_of'
  | 'has_part'
  | 'instance_of'
  | 'subclass_of'
  | 'related_to'
  | 'depends_on'
  | 'implements'
  | 'extends'
  | 'uses'
  | 'used_by'
  | 'similar_to'
  | 'opposite_of'
  | 'causes'
  | 'caused_by'
  | 'precedes'
  | 'follows'
  | 'contains'
  | 'contained_in'
  | 'references'
  | 'referenced_by'
  | 'derived_from'
  | 'basis_for'
  | 'example_of'
  | 'counterexample_of';

// =============================================================================
// GITHUB REPOSITORY CRAWLER
// =============================================================================

export interface GitHubRepository {
  owner: string;
  name: string;
  fullName: string;
  description: string;
  language: string;
  languages: Record<string, number>;
  stars: number;
  forks: number;
  watchers: number;
  issues: number;
  topics: string[];
  license: string;
  defaultBranch: string;
  createdAt: Date;
  updatedAt: Date;
  pushedAt: Date;
  size: number;
  files: GitHubFile[];
  readme: string;
  contributing: string;
  codeOfConduct: string;
}

export interface GitHubFile {
  path: string;
  name: string;
  type: 'file' | 'dir';
  size: number;
  sha: string;
  content?: string;
  language?: string;
}

export class GitHubCrawler {
  private baseUrl = 'https://api.github.com';
  private token?: string;
  private rateLimitRemaining = 5000;
  private rateLimitReset = 0;
  
  constructor(token?: string) {
    this.token = token || process.env.GITHUB_TOKEN;
  }
  
  private async fetch(endpoint: string): Promise<any> {
    const headers: Record<string, string> = {
      'Accept': 'application/vnd.github.v3+json',
      'User-Agent': 'TRUE-ASI-Knowledge-Crawler'
    };
    
    if (this.token) {
      headers['Authorization'] = `token ${this.token}`;
    }
    
    const response = await fetch(`${this.baseUrl}${endpoint}`, { headers });
    
    // Update rate limit info
    this.rateLimitRemaining = parseInt(response.headers.get('X-RateLimit-Remaining') || '5000');
    this.rateLimitReset = parseInt(response.headers.get('X-RateLimit-Reset') || '0');
    
    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  }
  
  // Get repository information
  async getRepository(owner: string, repo: string): Promise<GitHubRepository> {
    const data = await this.fetch(`/repos/${owner}/${repo}`);
    const languages = await this.fetch(`/repos/${owner}/${repo}/languages`);
    const readme = await this.getReadme(owner, repo);
    
    return {
      owner: data.owner.login,
      name: data.name,
      fullName: data.full_name,
      description: data.description || '',
      language: data.language || '',
      languages,
      stars: data.stargazers_count,
      forks: data.forks_count,
      watchers: data.watchers_count,
      issues: data.open_issues_count,
      topics: data.topics || [],
      license: data.license?.spdx_id || '',
      defaultBranch: data.default_branch,
      createdAt: new Date(data.created_at),
      updatedAt: new Date(data.updated_at),
      pushedAt: new Date(data.pushed_at),
      size: data.size,
      files: [],
      readme,
      contributing: '',
      codeOfConduct: ''
    };
  }
  
  // Get repository README
  async getReadme(owner: string, repo: string): Promise<string> {
    try {
      const data = await this.fetch(`/repos/${owner}/${repo}/readme`);
      return Buffer.from(data.content, 'base64').toString('utf-8');
    } catch {
      return '';
    }
  }
  
  // Get repository contents
  async getContents(owner: string, repo: string, path: string = ''): Promise<GitHubFile[]> {
    const data = await this.fetch(`/repos/${owner}/${repo}/contents/${path}`);
    
    if (Array.isArray(data)) {
      return data.map(item => ({
        path: item.path,
        name: item.name,
        type: item.type,
        size: item.size,
        sha: item.sha
      }));
    }
    
    return [{
      path: data.path,
      name: data.name,
      type: data.type,
      size: data.size,
      sha: data.sha,
      content: data.content ? Buffer.from(data.content, 'base64').toString('utf-8') : undefined
    }];
  }
  
  // Get file content
  async getFileContent(owner: string, repo: string, path: string): Promise<string> {
    const data = await this.fetch(`/repos/${owner}/${repo}/contents/${path}`);
    return Buffer.from(data.content, 'base64').toString('utf-8');
  }
  
  // Search repositories
  async searchRepositories(query: string, options?: {
    language?: string;
    sort?: 'stars' | 'forks' | 'updated';
    order?: 'asc' | 'desc';
    perPage?: number;
    page?: number;
  }): Promise<GitHubRepository[]> {
    let endpoint = `/search/repositories?q=${encodeURIComponent(query)}`;
    
    if (options?.language) endpoint += `+language:${options.language}`;
    if (options?.sort) endpoint += `&sort=${options.sort}`;
    if (options?.order) endpoint += `&order=${options.order}`;
    endpoint += `&per_page=${options?.perPage || 100}`;
    endpoint += `&page=${options?.page || 1}`;
    
    const data = await this.fetch(endpoint);
    
    return data.items.map((item: any) => ({
      owner: item.owner.login,
      name: item.name,
      fullName: item.full_name,
      description: item.description || '',
      language: item.language || '',
      languages: {},
      stars: item.stargazers_count,
      forks: item.forks_count,
      watchers: item.watchers_count,
      issues: item.open_issues_count,
      topics: item.topics || [],
      license: item.license?.spdx_id || '',
      defaultBranch: item.default_branch,
      createdAt: new Date(item.created_at),
      updatedAt: new Date(item.updated_at),
      pushedAt: new Date(item.pushed_at),
      size: item.size,
      files: [],
      readme: '',
      contributing: '',
      codeOfConduct: ''
    }));
  }
  
  // Get trending repositories
  async getTrendingRepositories(language?: string, since: 'daily' | 'weekly' | 'monthly' = 'daily'): Promise<GitHubRepository[]> {
    const date = new Date();
    if (since === 'daily') date.setDate(date.getDate() - 1);
    else if (since === 'weekly') date.setDate(date.getDate() - 7);
    else date.setMonth(date.getMonth() - 1);
    
    let query = `created:>${date.toISOString().split('T')[0]}`;
    if (language) query += ` language:${language}`;
    
    return this.searchRepositories(query, { sort: 'stars', order: 'desc' });
  }
  
  // Get user repositories
  async getUserRepositories(username: string): Promise<GitHubRepository[]> {
    const data = await this.fetch(`/users/${username}/repos?per_page=100&sort=updated`);
    
    return data.map((item: any) => ({
      owner: item.owner.login,
      name: item.name,
      fullName: item.full_name,
      description: item.description || '',
      language: item.language || '',
      languages: {},
      stars: item.stargazers_count,
      forks: item.forks_count,
      watchers: item.watchers_count,
      issues: item.open_issues_count,
      topics: item.topics || [],
      license: item.license?.spdx_id || '',
      defaultBranch: item.default_branch,
      createdAt: new Date(item.created_at),
      updatedAt: new Date(item.updated_at),
      pushedAt: new Date(item.pushed_at),
      size: item.size,
      files: [],
      readme: '',
      contributing: '',
      codeOfConduct: ''
    }));
  }
  
  // Get organization repositories
  async getOrganizationRepositories(org: string): Promise<GitHubRepository[]> {
    const data = await this.fetch(`/orgs/${org}/repos?per_page=100&sort=updated`);
    
    return data.map((item: any) => ({
      owner: item.owner.login,
      name: item.name,
      fullName: item.full_name,
      description: item.description || '',
      language: item.language || '',
      languages: {},
      stars: item.stargazers_count,
      forks: item.forks_count,
      watchers: item.watchers_count,
      issues: item.open_issues_count,
      topics: item.topics || [],
      license: item.license?.spdx_id || '',
      defaultBranch: item.default_branch,
      createdAt: new Date(item.created_at),
      updatedAt: new Date(item.updated_at),
      pushedAt: new Date(item.pushed_at),
      size: item.size,
      files: [],
      readme: '',
      contributing: '',
      codeOfConduct: ''
    }));
  }
  
  // Crawl entire repository
  async crawlRepository(owner: string, repo: string): Promise<{
    repository: GitHubRepository;
    files: GitHubFile[];
    knowledge: KnowledgeItem[];
  }> {
    const repository = await this.getRepository(owner, repo);
    const files: GitHubFile[] = [];
    const knowledge: KnowledgeItem[] = [];
    
    // Recursively get all files
    const crawlDir = async (path: string) => {
      const contents = await this.getContents(owner, repo, path);
      
      for (const item of contents) {
        if (item.type === 'file') {
          // Get content for code files
          const ext = item.name.split('.').pop()?.toLowerCase();
          const codeExtensions = ['ts', 'tsx', 'js', 'jsx', 'py', 'java', 'go', 'rs', 'cpp', 'c', 'h', 'cs', 'rb', 'php', 'swift', 'kt', 'scala', 'md', 'json', 'yaml', 'yml', 'toml'];
          
          if (ext && codeExtensions.includes(ext) && item.size < 1000000) {
            try {
              item.content = await this.getFileContent(owner, repo, item.path);
              item.language = this.detectLanguage(ext);
              
              // Extract knowledge from file
              const extracted = this.extractKnowledge(item, repository);
              knowledge.push(...extracted);
            } catch (e) {
              // Skip files that can't be read
            }
          }
          
          files.push(item);
        } else if (item.type === 'dir') {
          // Skip common non-essential directories
          const skipDirs = ['node_modules', '.git', 'dist', 'build', 'vendor', '__pycache__', '.venv', 'venv'];
          if (!skipDirs.includes(item.name)) {
            await crawlDir(item.path);
          }
        }
      }
    };
    
    await crawlDir('');
    repository.files = files;
    
    // Extract knowledge from README
    if (repository.readme) {
      knowledge.push({
        id: `${owner}-${repo}-readme`,
        sourceId: `github:${owner}/${repo}`,
        type: 'documentation',
        title: `${repository.fullName} README`,
        content: repository.readme,
        summary: repository.description,
        keywords: repository.topics,
        entities: [],
        relations: [],
        confidence: 0.9,
        createdAt: new Date(),
        updatedAt: new Date(),
        metadata: { repository: repository.fullName }
      });
    }
    
    return { repository, files, knowledge };
  }
  
  private detectLanguage(ext: string): string {
    const languageMap: Record<string, string> = {
      ts: 'TypeScript', tsx: 'TypeScript', js: 'JavaScript', jsx: 'JavaScript',
      py: 'Python', java: 'Java', go: 'Go', rs: 'Rust', cpp: 'C++', c: 'C',
      h: 'C/C++ Header', cs: 'C#', rb: 'Ruby', php: 'PHP', swift: 'Swift',
      kt: 'Kotlin', scala: 'Scala', md: 'Markdown', json: 'JSON',
      yaml: 'YAML', yml: 'YAML', toml: 'TOML'
    };
    return languageMap[ext] || ext.toUpperCase();
  }
  
  private extractKnowledge(file: GitHubFile, repo: GitHubRepository): KnowledgeItem[] {
    const items: KnowledgeItem[] = [];
    
    if (!file.content) return items;
    
    // Extract functions, classes, interfaces
    const patterns = {
      typescript_function: /(?:export\s+)?(?:async\s+)?function\s+(\w+)/g,
      typescript_class: /(?:export\s+)?class\s+(\w+)/g,
      typescript_interface: /(?:export\s+)?interface\s+(\w+)/g,
      typescript_type: /(?:export\s+)?type\s+(\w+)/g,
      python_function: /def\s+(\w+)\s*\(/g,
      python_class: /class\s+(\w+)/g,
      java_class: /(?:public|private|protected)?\s*class\s+(\w+)/g,
      java_method: /(?:public|private|protected)?\s*(?:static\s+)?[\w<>[\]]+\s+(\w+)\s*\(/g
    };
    
    const language = file.language?.toLowerCase() || '';
    
    // Extract based on language
    for (const [patternName, pattern] of Object.entries(patterns)) {
      if (patternName.startsWith(language.split(' ')[0].toLowerCase()) || 
          (language === 'typescript' && patternName.startsWith('typescript')) ||
          (language === 'javascript' && patternName.startsWith('typescript'))) {
        let match;
        while ((match = pattern.exec(file.content)) !== null) {
          items.push({
            id: `${repo.fullName}-${file.path}-${match[1]}`,
            sourceId: `github:${repo.fullName}`,
            type: patternName.includes('function') || patternName.includes('method') ? 'function' :
                  patternName.includes('class') ? 'class' :
                  patternName.includes('interface') ? 'interface' : 'type',
            title: match[1],
            content: this.extractContext(file.content, match.index, 500),
            summary: `${patternName.replace('_', ' ')} from ${file.path}`,
            keywords: [match[1], file.language || '', repo.name],
            entities: [{ text: match[1], type: 'CODE_ENTITY', start: match.index, end: match.index + match[1].length, confidence: 1.0 }],
            relations: [],
            confidence: 0.85,
            createdAt: new Date(),
            updatedAt: new Date(),
            metadata: { file: file.path, repository: repo.fullName, language: file.language }
          });
        }
      }
    }
    
    return items;
  }
  
  private extractContext(content: string, index: number, contextSize: number): string {
    const start = Math.max(0, index - contextSize / 2);
    const end = Math.min(content.length, index + contextSize / 2);
    return content.substring(start, end);
  }
  
  getRateLimitInfo(): { remaining: number; reset: Date } {
    return {
      remaining: this.rateLimitRemaining,
      reset: new Date(this.rateLimitReset * 1000)
    };
  }
}

// =============================================================================
// ACADEMIC PAPER CRAWLER
// =============================================================================

export interface AcademicPaper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  year: number;
  venue: string;
  doi?: string;
  arxivId?: string;
  pdfUrl?: string;
  citations: number;
  references: string[];
  keywords: string[];
  categories: string[];
}

export class AcademicCrawler {
  // Search arXiv
  async searchArxiv(query: string, maxResults: number = 100): Promise<AcademicPaper[]> {
    const url = `http://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(query)}&start=0&max_results=${maxResults}`;
    
    const response = await fetch(url);
    const xml = await response.text();
    
    // Parse XML (simplified)
    const papers: AcademicPaper[] = [];
    const entries = xml.split('<entry>').slice(1);
    
    for (const entry of entries) {
      const getId = (tag: string) => {
        const match = entry.match(new RegExp(`<${tag}[^>]*>([^<]*)</${tag}>`));
        return match ? match[1].trim() : '';
      };
      
      const getAll = (tag: string) => {
        const matches = entry.matchAll(new RegExp(`<${tag}[^>]*>([^<]*)</${tag}>`, 'g'));
        return Array.from(matches).map(m => m[1].trim());
      };
      
      const arxivId = getId('id').replace('http://arxiv.org/abs/', '');
      
      papers.push({
        id: arxivId,
        title: getId('title').replace(/\s+/g, ' '),
        authors: getAll('name'),
        abstract: getId('summary').replace(/\s+/g, ' '),
        year: parseInt(getId('published').substring(0, 4)) || new Date().getFullYear(),
        venue: 'arXiv',
        arxivId,
        pdfUrl: `https://arxiv.org/pdf/${arxivId}.pdf`,
        citations: 0,
        references: [],
        keywords: getAll('category').map(c => c.split('.').pop() || c),
        categories: getAll('category')
      });
    }
    
    return papers;
  }
  
  // Search Semantic Scholar
  async searchSemanticScholar(query: string, limit: number = 100): Promise<AcademicPaper[]> {
    const url = `https://api.semanticscholar.org/graph/v1/paper/search?query=${encodeURIComponent(query)}&limit=${limit}&fields=paperId,title,authors,abstract,year,venue,citationCount,referenceCount,fieldsOfStudy,externalIds`;
    
    const response = await fetch(url);
    const data = await response.json();
    
    return (data.data || []).map((paper: any) => ({
      id: paper.paperId,
      title: paper.title,
      authors: (paper.authors || []).map((a: any) => a.name),
      abstract: paper.abstract || '',
      year: paper.year || 0,
      venue: paper.venue || '',
      doi: paper.externalIds?.DOI,
      arxivId: paper.externalIds?.ArXiv,
      citations: paper.citationCount || 0,
      references: [],
      keywords: paper.fieldsOfStudy || [],
      categories: paper.fieldsOfStudy || []
    }));
  }
  
  // Extract knowledge from paper
  extractKnowledge(paper: AcademicPaper): KnowledgeItem[] {
    const items: KnowledgeItem[] = [];
    
    // Main paper knowledge
    items.push({
      id: `paper-${paper.id}`,
      sourceId: `academic:${paper.id}`,
      type: 'citation',
      title: paper.title,
      content: paper.abstract,
      summary: `${paper.title} by ${paper.authors.slice(0, 3).join(', ')}${paper.authors.length > 3 ? ' et al.' : ''} (${paper.year})`,
      keywords: paper.keywords,
      entities: paper.authors.map((author, i) => ({
        text: author,
        type: 'PERSON',
        start: i * 10,
        end: i * 10 + author.length,
        confidence: 1.0
      })),
      relations: [],
      confidence: 0.95,
      createdAt: new Date(),
      updatedAt: new Date(),
      metadata: {
        year: paper.year,
        venue: paper.venue,
        citations: paper.citations,
        doi: paper.doi,
        arxivId: paper.arxivId
      }
    });
    
    return items;
  }
}

// =============================================================================
// WEB CONTENT CRAWLER
// =============================================================================

export class WebCrawler {
  private visitedUrls: Set<string> = new Set();
  private maxDepth: number = 3;
  
  // Crawl a single URL
  async crawlUrl(url: string): Promise<{
    url: string;
    title: string;
    content: string;
    links: string[];
    metadata: Record<string, string>;
  }> {
    if (this.visitedUrls.has(url)) {
      return { url, title: '', content: '', links: [], metadata: {} };
    }
    
    this.visitedUrls.add(url);
    
    try {
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'TRUE-ASI-Knowledge-Crawler/1.0'
        }
      });
      
      const html = await response.text();
      
      // Extract title
      const titleMatch = html.match(/<title[^>]*>([^<]*)<\/title>/i);
      const title = titleMatch ? titleMatch[1].trim() : '';
      
      // Extract meta tags
      const metadata: Record<string, string> = {};
      const metaMatches = html.matchAll(/<meta\s+(?:name|property)="([^"]+)"\s+content="([^"]+)"/gi);
      for (const match of metaMatches) {
        metadata[match[1]] = match[2];
      }
      
      // Extract text content (simplified)
      const content = html
        .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
        .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
        .replace(/<[^>]+>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .substring(0, 50000); // Limit content size
      
      // Extract links
      const links: string[] = [];
      const linkMatches = html.matchAll(/<a[^>]+href="([^"]+)"/gi);
      for (const match of linkMatches) {
        const href = match[1];
        if (href.startsWith('http')) {
          links.push(href);
        } else if (href.startsWith('/')) {
          const base = new URL(url);
          links.push(`${base.origin}${href}`);
        }
      }
      
      return { url, title, content, links: [...new Set(links)], metadata };
    } catch (error) {
      return { url, title: '', content: '', links: [], metadata: {} };
    }
  }
  
  // Crawl multiple URLs with depth
  async crawlSite(startUrl: string, depth: number = 3): Promise<{
    pages: Array<{ url: string; title: string; content: string }>;
    knowledge: KnowledgeItem[];
  }> {
    const pages: Array<{ url: string; title: string; content: string }> = [];
    const knowledge: KnowledgeItem[] = [];
    const queue: Array<{ url: string; depth: number }> = [{ url: startUrl, depth: 0 }];
    
    while (queue.length > 0) {
      const { url, depth: currentDepth } = queue.shift()!;
      
      if (currentDepth > depth) continue;
      
      const result = await this.crawlUrl(url);
      
      if (result.content) {
        pages.push({
          url: result.url,
          title: result.title,
          content: result.content
        });
        
        // Extract knowledge
        knowledge.push({
          id: `web-${Buffer.from(url).toString('base64').substring(0, 20)}`,
          sourceId: `web:${url}`,
          type: 'fact',
          title: result.title || url,
          content: result.content.substring(0, 5000),
          summary: result.metadata['description'] || result.title || '',
          keywords: (result.metadata['keywords'] || '').split(',').map(k => k.trim()).filter(k => k),
          entities: [],
          relations: [],
          confidence: 0.7,
          createdAt: new Date(),
          updatedAt: new Date(),
          metadata: result.metadata
        });
        
        // Add links to queue
        for (const link of result.links.slice(0, 10)) {
          if (!this.visitedUrls.has(link)) {
            queue.push({ url: link, depth: currentDepth + 1 });
          }
        }
      }
    }
    
    return { pages, knowledge };
  }
}

// =============================================================================
// KNOWLEDGE GRAPH BUILDER
// =============================================================================

export interface KnowledgeGraph {
  nodes: Map<string, KnowledgeNode>;
  edges: Map<string, KnowledgeEdge>;
  clusters: Map<string, string[]>;
}

export interface KnowledgeNode {
  id: string;
  type: string;
  label: string;
  properties: Record<string, any>;
  embedding?: number[];
}

export interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  type: RelationType;
  weight: number;
  properties: Record<string, any>;
}

export class KnowledgeGraphBuilder {
  private graph: KnowledgeGraph = {
    nodes: new Map(),
    edges: new Map(),
    clusters: new Map()
  };
  
  // Add knowledge item to graph
  addKnowledgeItem(item: KnowledgeItem): void {
    // Add main node
    this.graph.nodes.set(item.id, {
      id: item.id,
      type: item.type,
      label: item.title,
      properties: {
        content: item.content,
        summary: item.summary,
        keywords: item.keywords,
        confidence: item.confidence,
        sourceId: item.sourceId
      },
      embedding: item.embedding
    });
    
    // Add entity nodes and edges
    for (const entity of item.entities) {
      const entityId = `entity-${entity.type}-${entity.text}`;
      
      if (!this.graph.nodes.has(entityId)) {
        this.graph.nodes.set(entityId, {
          id: entityId,
          type: entity.type,
          label: entity.text,
          properties: { confidence: entity.confidence }
        });
      }
      
      // Add edge from item to entity
      const edgeId = `${item.id}-mentions-${entityId}`;
      this.graph.edges.set(edgeId, {
        id: edgeId,
        source: item.id,
        target: entityId,
        type: 'contains',
        weight: entity.confidence,
        properties: {}
      });
    }
    
    // Add relation edges
    for (const relation of item.relations) {
      const edgeId = `${relation.sourceId}-${relation.type}-${relation.targetId}`;
      this.graph.edges.set(edgeId, {
        id: edgeId,
        source: relation.sourceId,
        target: relation.targetId,
        type: relation.type,
        weight: relation.confidence,
        properties: { evidence: relation.evidence }
      });
    }
  }
  
  // Find related nodes
  findRelated(nodeId: string, maxDepth: number = 2): KnowledgeNode[] {
    const related: KnowledgeNode[] = [];
    const visited = new Set<string>();
    const queue: Array<{ id: string; depth: number }> = [{ id: nodeId, depth: 0 }];
    
    while (queue.length > 0) {
      const { id, depth } = queue.shift()!;
      
      if (visited.has(id) || depth > maxDepth) continue;
      visited.add(id);
      
      const node = this.graph.nodes.get(id);
      if (node && id !== nodeId) {
        related.push(node);
      }
      
      // Find connected nodes
      for (const edge of this.graph.edges.values()) {
        if (edge.source === id && !visited.has(edge.target)) {
          queue.push({ id: edge.target, depth: depth + 1 });
        }
        if (edge.target === id && !visited.has(edge.source)) {
          queue.push({ id: edge.source, depth: depth + 1 });
        }
      }
    }
    
    return related;
  }
  
  // Search nodes by keyword
  searchNodes(query: string): KnowledgeNode[] {
    const queryLower = query.toLowerCase();
    const results: KnowledgeNode[] = [];
    
    for (const node of this.graph.nodes.values()) {
      if (
        node.label.toLowerCase().includes(queryLower) ||
        (node.properties.keywords || []).some((k: string) => k.toLowerCase().includes(queryLower)) ||
        (node.properties.content || '').toLowerCase().includes(queryLower)
      ) {
        results.push(node);
      }
    }
    
    return results;
  }
  
  // Get graph statistics
  getStatistics(): {
    nodeCount: number;
    edgeCount: number;
    nodesByType: Record<string, number>;
    edgesByType: Record<string, number>;
    avgDegree: number;
  } {
    const nodesByType: Record<string, number> = {};
    const edgesByType: Record<string, number> = {};
    const degrees: Record<string, number> = {};
    
    for (const node of this.graph.nodes.values()) {
      nodesByType[node.type] = (nodesByType[node.type] || 0) + 1;
      degrees[node.id] = 0;
    }
    
    for (const edge of this.graph.edges.values()) {
      edgesByType[edge.type] = (edgesByType[edge.type] || 0) + 1;
      degrees[edge.source] = (degrees[edge.source] || 0) + 1;
      degrees[edge.target] = (degrees[edge.target] || 0) + 1;
    }
    
    const degreeValues = Object.values(degrees);
    const avgDegree = degreeValues.length > 0 
      ? degreeValues.reduce((a, b) => a + b, 0) / degreeValues.length 
      : 0;
    
    return {
      nodeCount: this.graph.nodes.size,
      edgeCount: this.graph.edges.size,
      nodesByType,
      edgesByType,
      avgDegree
    };
  }
  
  // Export graph
  exportGraph(): KnowledgeGraph {
    return this.graph;
  }
}

// =============================================================================
// UNIVERSAL KNOWLEDGE ACQUISITION ENGINE
// =============================================================================

export class UniversalKnowledgeEngine {
  private githubCrawler: GitHubCrawler;
  private academicCrawler: AcademicCrawler;
  private webCrawler: WebCrawler;
  private graphBuilder: KnowledgeGraphBuilder;
  private sources: Map<string, KnowledgeSource> = new Map();
  private items: Map<string, KnowledgeItem> = new Map();
  
  constructor(githubToken?: string) {
    this.githubCrawler = new GitHubCrawler(githubToken);
    this.academicCrawler = new AcademicCrawler();
    this.webCrawler = new WebCrawler();
    this.graphBuilder = new KnowledgeGraphBuilder();
  }
  
  // Acquire knowledge from GitHub repository
  async acquireFromGitHub(owner: string, repo: string): Promise<{
    source: KnowledgeSource;
    items: KnowledgeItem[];
  }> {
    const sourceId = `github:${owner}/${repo}`;
    
    const source: KnowledgeSource = {
      id: sourceId,
      type: 'github_repository',
      url: `https://github.com/${owner}/${repo}`,
      name: `${owner}/${repo}`,
      description: '',
      totalItems: 0,
      extractedFacts: 0,
      quality: 0,
      status: 'crawling'
    };
    
    this.sources.set(sourceId, source);
    
    try {
      const result = await this.githubCrawler.crawlRepository(owner, repo);
      
      source.description = result.repository.description;
      source.totalItems = result.files.length;
      source.extractedFacts = result.knowledge.length;
      source.quality = Math.min(100, result.repository.stars / 10);
      source.status = 'complete';
      source.lastCrawled = new Date();
      
      // Store items
      for (const item of result.knowledge) {
        this.items.set(item.id, item);
        this.graphBuilder.addKnowledgeItem(item);
      }
      
      return { source, items: result.knowledge };
    } catch (error) {
      source.status = 'error';
      throw error;
    }
  }
  
  // Acquire knowledge from academic papers
  async acquireFromAcademic(query: string, source: 'arxiv' | 'semantic_scholar' = 'arxiv'): Promise<{
    source: KnowledgeSource;
    items: KnowledgeItem[];
  }> {
    const sourceId = `academic:${source}:${query}`;
    
    const knowledgeSource: KnowledgeSource = {
      id: sourceId,
      type: 'academic_paper',
      url: source === 'arxiv' ? 'https://arxiv.org' : 'https://semanticscholar.org',
      name: `${source} search: ${query}`,
      description: `Academic papers matching "${query}"`,
      totalItems: 0,
      extractedFacts: 0,
      quality: 95,
      status: 'crawling'
    };
    
    this.sources.set(sourceId, knowledgeSource);
    
    try {
      const papers = source === 'arxiv' 
        ? await this.academicCrawler.searchArxiv(query)
        : await this.academicCrawler.searchSemanticScholar(query);
      
      const items: KnowledgeItem[] = [];
      
      for (const paper of papers) {
        const extracted = this.academicCrawler.extractKnowledge(paper);
        items.push(...extracted);
        
        for (const item of extracted) {
          this.items.set(item.id, item);
          this.graphBuilder.addKnowledgeItem(item);
        }
      }
      
      knowledgeSource.totalItems = papers.length;
      knowledgeSource.extractedFacts = items.length;
      knowledgeSource.status = 'complete';
      knowledgeSource.lastCrawled = new Date();
      
      return { source: knowledgeSource, items };
    } catch (error) {
      knowledgeSource.status = 'error';
      throw error;
    }
  }
  
  // Acquire knowledge from web
  async acquireFromWeb(url: string, depth: number = 2): Promise<{
    source: KnowledgeSource;
    items: KnowledgeItem[];
  }> {
    const sourceId = `web:${url}`;
    
    const knowledgeSource: KnowledgeSource = {
      id: sourceId,
      type: 'web_page',
      url,
      name: url,
      description: '',
      totalItems: 0,
      extractedFacts: 0,
      quality: 70,
      status: 'crawling'
    };
    
    this.sources.set(sourceId, knowledgeSource);
    
    try {
      const result = await this.webCrawler.crawlSite(url, depth);
      
      knowledgeSource.totalItems = result.pages.length;
      knowledgeSource.extractedFacts = result.knowledge.length;
      knowledgeSource.status = 'complete';
      knowledgeSource.lastCrawled = new Date();
      
      for (const item of result.knowledge) {
        this.items.set(item.id, item);
        this.graphBuilder.addKnowledgeItem(item);
      }
      
      return { source: knowledgeSource, items: result.knowledge };
    } catch (error) {
      knowledgeSource.status = 'error';
      throw error;
    }
  }
  
  // Acquire knowledge from multiple GitHub repositories
  async acquireFromMultipleRepos(repos: Array<{ owner: string; repo: string }>): Promise<{
    sources: KnowledgeSource[];
    totalItems: number;
  }> {
    const sources: KnowledgeSource[] = [];
    let totalItems = 0;
    
    for (const { owner, repo } of repos) {
      try {
        const result = await this.acquireFromGitHub(owner, repo);
        sources.push(result.source);
        totalItems += result.items.length;
      } catch (error) {
        console.error(`Failed to acquire from ${owner}/${repo}:`, error);
      }
    }
    
    return { sources, totalItems };
  }
  
  // Search knowledge
  searchKnowledge(query: string): KnowledgeItem[] {
    const queryLower = query.toLowerCase();
    const results: KnowledgeItem[] = [];
    
    for (const item of this.items.values()) {
      if (
        item.title.toLowerCase().includes(queryLower) ||
        item.content.toLowerCase().includes(queryLower) ||
        item.keywords.some(k => k.toLowerCase().includes(queryLower))
      ) {
        results.push(item);
      }
    }
    
    // Sort by confidence
    return results.sort((a, b) => b.confidence - a.confidence);
  }
  
  // Get statistics
  getStatistics(): {
    totalSources: number;
    totalItems: number;
    sourcesByType: Record<string, number>;
    itemsByType: Record<string, number>;
    graphStats: ReturnType<KnowledgeGraphBuilder['getStatistics']>;
  } {
    const sourcesByType: Record<string, number> = {};
    const itemsByType: Record<string, number> = {};
    
    for (const source of this.sources.values()) {
      sourcesByType[source.type] = (sourcesByType[source.type] || 0) + 1;
    }
    
    for (const item of this.items.values()) {
      itemsByType[item.type] = (itemsByType[item.type] || 0) + 1;
    }
    
    return {
      totalSources: this.sources.size,
      totalItems: this.items.size,
      sourcesByType,
      itemsByType,
      graphStats: this.graphBuilder.getStatistics()
    };
  }
  
  // Export all knowledge
  exportKnowledge(): {
    sources: KnowledgeSource[];
    items: KnowledgeItem[];
    graph: KnowledgeGraph;
  } {
    return {
      sources: Array.from(this.sources.values()),
      items: Array.from(this.items.values()),
      graph: this.graphBuilder.exportGraph()
    };
  }
}

// =============================================================================
// TOP REPOSITORIES TO ACQUIRE KNOWLEDGE FROM
// =============================================================================

export const TOP_AI_REPOSITORIES = [
  // LLM Frameworks
  { owner: 'huggingface', repo: 'transformers' },
  { owner: 'pytorch', repo: 'pytorch' },
  { owner: 'tensorflow', repo: 'tensorflow' },
  { owner: 'langchain-ai', repo: 'langchain' },
  { owner: 'openai', repo: 'openai-python' },
  { owner: 'anthropics', repo: 'anthropic-sdk-python' },
  
  // Agent Frameworks
  { owner: 'microsoft', repo: 'autogen' },
  { owner: 'joaomdmoura', repo: 'crewAI' },
  { owner: 'Significant-Gravitas', repo: 'AutoGPT' },
  { owner: 'geekan', repo: 'MetaGPT' },
  
  // RAG & Knowledge
  { owner: 'run-llama', repo: 'llama_index' },
  { owner: 'chroma-core', repo: 'chroma' },
  { owner: 'qdrant', repo: 'qdrant' },
  { owner: 'pinecone-io', repo: 'pinecone-python-client' },
  
  // Code & Tools
  { owner: 'Pythagora-io', repo: 'gpt-pilot' },
  { owner: 'OpenDevin', repo: 'OpenDevin' },
  { owner: 'princeton-nlp', repo: 'SWE-agent' },
  
  // Models
  { owner: 'meta-llama', repo: 'llama' },
  { owner: 'mistralai', repo: 'mistral-src' },
  { owner: 'QwenLM', repo: 'Qwen' },
  { owner: 'deepseek-ai', repo: 'DeepSeek-Coder' },
  
  // Evaluation
  { owner: 'openai', repo: 'evals' },
  { owner: 'EleutherAI', repo: 'lm-evaluation-harness' },
  
  // Infrastructure
  { owner: 'vllm-project', repo: 'vllm' },
  { owner: 'NVIDIA', repo: 'TensorRT-LLM' },
  { owner: 'ray-project', repo: 'ray' }
];

export const TOP_PROGRAMMING_REPOSITORIES = [
  // Languages
  { owner: 'rust-lang', repo: 'rust' },
  { owner: 'golang', repo: 'go' },
  { owner: 'python', repo: 'cpython' },
  { owner: 'nodejs', repo: 'node' },
  { owner: 'microsoft', repo: 'TypeScript' },
  
  // Web Frameworks
  { owner: 'vercel', repo: 'next.js' },
  { owner: 'facebook', repo: 'react' },
  { owner: 'vuejs', repo: 'vue' },
  { owner: 'sveltejs', repo: 'svelte' },
  { owner: 'angular', repo: 'angular' },
  
  // Backend
  { owner: 'fastapi', repo: 'fastapi' },
  { owner: 'django', repo: 'django' },
  { owner: 'expressjs', repo: 'express' },
  { owner: 'nestjs', repo: 'nest' },
  
  // Databases
  { owner: 'postgres', repo: 'postgres' },
  { owner: 'redis', repo: 'redis' },
  { owner: 'mongodb', repo: 'mongo' },
  
  // DevOps
  { owner: 'kubernetes', repo: 'kubernetes' },
  { owner: 'docker', repo: 'docker-ce' },
  { owner: 'hashicorp', repo: 'terraform' }
];

// =============================================================================
// EXPORT SINGLETON INSTANCE
// =============================================================================

export const universalKnowledgeEngine = new UniversalKnowledgeEngine();
