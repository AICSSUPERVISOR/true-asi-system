# Contributing to TRUE ASI System

Thank you for your interest in contributing to the TRUE ASI System! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Code Style Guide](#code-style-guide)
5. [Commit Message Guidelines](#commit-message-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Testing Guidelines](#testing-guidelines)
8. [Documentation Standards](#documentation-standards)

---

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Prioritize code quality and maintainability
- Follow security best practices
- Document your changes thoroughly

---

## Getting Started

### Prerequisites

- Node.js 22.13.0 or higher
- pnpm 10.4.1 or higher
- MySQL/TiDB database access
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/AICSSUPERVISOR/true-asi-system.git
cd true-asi-system

# Install dependencies
pnpm install

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Push database schema
pnpm db:push

# Start development server
pnpm dev
```

---

## Development Workflow

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions or fixes
- `chore/description` - Maintenance tasks

### Example

```bash
git checkout -b feature/add-export-scheduler
```

---

## Code Style Guide

### TypeScript

- Use TypeScript for all new code
- Enable strict mode
- Avoid `any` types - use `unknown` or proper types
- Use interfaces for object shapes
- Use type aliases for unions and complex types

```typescript
// ✅ Good
interface User {
  id: number;
  name: string;
  email: string | null;
}

// ❌ Bad
const user: any = { id: 1, name: "John" };
```

### React Components

- Use functional components with hooks
- Prefer named exports over default exports
- Keep components small and focused (< 200 lines)
- Extract reusable logic into custom hooks
- Use TypeScript for prop types

```typescript
// ✅ Good
export function UserProfile({ userId }: { userId: number }) {
  const { data, isLoading } = trpc.user.getById.useQuery({ userId });
  
  if (isLoading) return <LoadingSkeleton variant="card" />;
  
  return <div>{data?.name}</div>;
}

// ❌ Bad
export default function UserProfile(props: any) {
  // ...
}
```

### tRPC Procedures

- Use Zod for input validation
- Return typed responses
- Handle errors with `TRPCError`
- Use `protectedProcedure` for authenticated endpoints

```typescript
// ✅ Good
getUser: protectedProcedure
  .input(z.object({ userId: z.number() }))
  .query(async ({ ctx, input }) => {
    const db = await getDb();
    if (!db) throw new TRPCError({ code: 'INTERNAL_SERVER_ERROR' });
    
    const user = await db.query.users.findFirst({
      where: eq(users.id, input.userId),
    });
    
    if (!user) throw new TRPCError({ code: 'NOT_FOUND' });
    return user;
  }),
```

### Database Queries

- Always use `await getDb()` pattern
- Check if `db` is null before querying
- Use Drizzle ORM for type safety
- Avoid raw SQL queries unless necessary

```typescript
// ✅ Good
const db = await getDb();
if (!db) return { success: false };

const users = await db.select().from(usersTable).where(eq(usersTable.id, userId));

// ❌ Bad
const users = await db.query(`SELECT * FROM users WHERE id = ${userId}`);
```

### Styling

- Use Tailwind CSS utilities
- Follow the design system in `client/src/index.css`
- Use shadcn/ui components for consistency
- Avoid inline styles

```tsx
// ✅ Good
<div className="bg-white/5 backdrop-blur-xl border-white/10 rounded-lg p-6">
  <h2 className="text-2xl font-bold text-white mb-4">Title</h2>
</div>

// ❌ Bad
<div style={{ background: 'rgba(255,255,255,0.05)', padding: '24px' }}>
  <h2 style={{ fontSize: '24px', fontWeight: 'bold' }}>Title</h2>
</div>
```

---

## Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(notifications): add real-time notification center

- Created NotificationCenter dropdown component
- Integrated WebSocket for real-time updates
- Added read/unread status tracking

Closes #123
```

```
fix(revenue-tracking): correct CSV export date formatting

The CSV export was using incorrect date format. Changed to ISO 8601.

Fixes #456
```

---

## Pull Request Process

### Before Submitting

1. **Run tests**: `pnpm test`
2. **Check TypeScript**: `pnpm type-check`
3. **Lint code**: `pnpm lint`
4. **Format code**: `pnpm format`
5. **Update documentation**: Add/update relevant docs
6. **Update CHANGELOG.md**: Add entry under `[Unreleased]`

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests pass locally
- [ ] CHANGELOG.md updated
```

### Review Process

1. Submit PR with clear description
2. Address reviewer feedback
3. Ensure CI/CD passes
4. Get approval from at least one maintainer
5. Squash and merge

---

## Testing Guidelines

### Unit Tests

- Test all tRPC procedures
- Use Vitest for testing
- Mock external dependencies
- Aim for 80%+ code coverage

```typescript
// Example: server/notifications.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { appRouter } from './routers';

describe('notifications.getAll', () => {
  it('should return all notifications for authenticated user', async () => {
    const caller = appRouter.createCaller({ user: mockUser });
    const result = await caller.notifications.getAll();
    
    expect(result).toHaveLength(5);
    expect(result[0]).toHaveProperty('title');
  });
});
```

### Integration Tests

- Test critical user flows
- Use real database (test environment)
- Clean up test data after each test

### Manual Testing

- Test in multiple browsers (Chrome, Firefox, Safari)
- Test responsive design (mobile, tablet, desktop)
- Test accessibility (keyboard navigation, screen readers)

---

## Documentation Standards

### Code Comments

- Use JSDoc for functions and types
- Explain "why", not "what"
- Keep comments up-to-date

```typescript
/**
 * Emit real-time metric update event to subscribed clients
 * 
 * @param analysisId - Unique identifier for the analysis
 * @param metrics - Updated metric values
 * 
 * @example
 * emitMetricUpdate('analysis_123', { revenue: 50000, customers: 120 });
 */
export function emitMetricUpdate(analysisId: string, metrics: any) {
  // Implementation
}
```

### README Updates

- Update README.md when adding new features
- Include setup instructions for new dependencies
- Add examples for new APIs

### API Documentation

- Document all tRPC procedures
- Include input/output types
- Provide usage examples

---

## Questions?

If you have questions or need help, please:

1. Check existing documentation
2. Search closed issues
3. Open a new issue with the `question` label
4. Contact maintainers via email

---

## License

By contributing to TRUE ASI System, you agree that your contributions will be licensed under the project's proprietary license.

---

**Thank you for contributing to TRUE ASI System!** 🚀
