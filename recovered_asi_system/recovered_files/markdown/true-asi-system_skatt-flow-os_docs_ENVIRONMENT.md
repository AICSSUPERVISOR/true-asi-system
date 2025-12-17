# Skatt-Flow OS - Environment Variables Documentation

This document describes all environment variables required for Skatt-Flow OS to function properly.

## System Variables (Auto-configured)

These variables are automatically configured by the Manus platform:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | MySQL/TiDB connection string |
| `JWT_SECRET` | Session cookie signing secret |
| `VITE_APP_ID` | Manus OAuth application ID |
| `OAUTH_SERVER_URL` | Manus OAuth backend base URL |
| `VITE_OAUTH_PORTAL_URL` | Manus login portal URL (frontend) |
| `OWNER_OPEN_ID` | Owner's Manus OpenID |
| `OWNER_NAME` | Owner's display name |
| `BUILT_IN_FORGE_API_URL` | Manus built-in APIs URL |
| `BUILT_IN_FORGE_API_KEY` | Bearer token for Manus APIs (server-side) |
| `VITE_FRONTEND_FORGE_API_KEY` | Bearer token for frontend Manus APIs |
| `VITE_FRONTEND_FORGE_API_URL` | Manus APIs URL for frontend |

## External API Integrations

### Forvalt / Proff API

For company data enrichment and credit scoring:

| Variable | Description | Required |
|----------|-------------|----------|
| `FORVALT_API_KEY` | API key for Forvalt/Proff services | Yes |
| `FORVALT_API_URL` | Base URL (default: `https://api.forvalt.no`) | No |

### Altinn Integration

For filing MVA-meldinger and other regulatory submissions:

| Variable | Description | Required |
|----------|-------------|----------|
| `ALTINN_CLIENT_ID` | OAuth2 client ID for Altinn | Yes |
| `ALTINN_CLIENT_SECRET` | OAuth2 client secret | Yes |
| `ALTINN_ENVIRONMENT` | Environment (`production` or `test`) | No |
| `ALTINN_SCOPES` | OAuth2 scopes (comma-separated) | No |

### Accounting System Integrations

#### Tripletex

| Variable | Description | Required |
|----------|-------------|----------|
| `TRIPLETEX_CONSUMER_TOKEN` | Consumer token for Tripletex API | Yes |
| `TRIPLETEX_EMPLOYEE_TOKEN` | Employee token | Yes |
| `TRIPLETEX_API_URL` | Base URL (default: `https://tripletex.no/v2`) | No |

#### PowerOffice

| Variable | Description | Required |
|----------|-------------|----------|
| `POWEROFFICE_CLIENT_KEY` | Client key for PowerOffice Go | Yes |
| `POWEROFFICE_API_URL` | Base URL | No |

#### Fiken

| Variable | Description | Required |
|----------|-------------|----------|
| `FIKEN_ACCESS_TOKEN` | OAuth2 access token for Fiken | Yes |
| `FIKEN_API_URL` | Base URL (default: `https://api.fiken.no/api/v2`) | No |

#### Visma eAccounting

| Variable | Description | Required |
|----------|-------------|----------|
| `VISMA_CLIENT_ID` | OAuth2 client ID | Yes |
| `VISMA_CLIENT_SECRET` | OAuth2 client secret | Yes |
| `VISMA_API_URL` | Base URL | No |

### AI/ML API

For document processing and intelligent suggestions:

| Variable | Description | Required |
|----------|-------------|----------|
| `AIML_API_KEY` | API key for AIML multi-model API | Yes |
| `AIML_API_URL` | Base URL (default: `https://api.aimlapi.com/v1`) | No |

## Feature Flags

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_AUTO_POST` | Enable automatic posting of AI-processed documents | `false` |
| `ENABLE_ALTINN_PRODUCTION` | Use Altinn production environment | `false` |
| `LOG_API_CALLS` | Log all external API calls (no secrets) | `true` |

## Security Notes

1. **Never commit secrets to version control** - Use environment variables or secrets management
2. **Rotate API keys regularly** - Especially for production environments
3. **Use separate credentials** for development, staging, and production
4. **Altinn production access** requires formal approval from Digitaliseringsdirektoratet

## Development Setup

For local development, create a `.env` file in the project root:

```bash
# Copy from .env.example
cp .env.example .env

# Edit with your credentials
nano .env
```

## Production Deployment

For production, configure environment variables through your hosting platform's secrets management:

- **Manus Platform**: Use the Settings → Secrets panel
- **Docker**: Use Docker secrets or environment files
- **Kubernetes**: Use ConfigMaps and Secrets

## Validation

The application validates required environment variables on startup. Missing critical variables will prevent the server from starting.

To test your configuration:

```bash
pnpm run check
```

This will verify TypeScript types and check for common configuration issues.
