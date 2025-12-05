# Changelog

All notable changes to the TRUE ASI System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Input sanitization middleware for XSS protection
- Database indexes for performance optimization (users table)
- WebSocket authentication via session cookies
- Comprehensive security documentation

## [1.3.0] - 2025-12-05

### Added
- **Backend WebSocket Implementation**: Real-time event emitters for metric updates, execution progress, and analysis completion
- **Notification Center**: Full-featured dropdown with real-time alerts, read/unread tracking, and color-coded notification types
- **ConnectionStatus Indicator**: Live WebSocket connection status with pulse animation across all dashboards
- **Real-Time Metric Updates**: `useRealtimeMetrics` and `useRealtimeExecution` hooks for live data refreshes
- **Database Schema**: Added `notifications`, `scheduled_exports`, and `export_history` tables
- **tRPC Procedures**: 5 new notification procedures (getAll, getUnreadCount, markAsRead, markAllAsRead, delete)

### Changed
- Enhanced `server/_core/websocket.ts` with TRUE ASI event emitters
- Updated all dashboard headers to include NotificationCenter and ConnectionStatus
- Improved WebSocket connection management with auto-reconnection

### Fixed
- TypeScript compilation errors in notification router
- WebSocket event subscription/unsubscription logic

## [1.2.0] - 2025-12-04

### Added
- **Premium UI Enhancements**: Applied glass-morphism design across all 6 dashboards
- **Loading Skeletons**: Created `LoadingSkeleton` component with 5 variants (card, chart, table, metric, list)
- **CSV Export**: Added downloadable CSV export to RevenueTracking dashboard with 9 metrics
- **Multi-Model Consensus**: Implemented parallel AI querying with 3-5 models for superhuman intelligence
- **Deeplink Expansion**: Documented 500+ platforms across 50 industries in `DEEPLINK_EXPANSION_500_PLUS.md`

### Changed
- Enhanced typography with `text-5xl font-black tracking-tight` for headers
- Applied backdrop-blur-xl, gradient overlays, and multi-layer shadows to all cards
- Improved hover effects with `hover:scale-[1.02]` and shadow animations

### Performance
- Integrated loading skeletons into RevenueTracking and AnalysisHistory for perceived performance

## [1.1.0] - 2025-12-03

### Added
- **Revenue Tracking Dashboard**: Complete dashboard with 6 metric cards, line charts, and time range filters
- **Analysis History Dashboard**: Comprehensive history view with search, filters, and bulk actions
- **Execution Dashboard**: Real-time automation progress tracking with task breakdown
- **Business Analysis System**: Multi-model AI consensus for business recommendations
- **Deeplink Registry**: 300+ platforms coded in `server/helpers/industry_deeplinks.ts`

### Changed
- Upgraded to React 19 and Tailwind 4
- Migrated from Axios to tRPC for type-safe API calls
- Implemented Drizzle ORM for database operations

### Security
- Added Helmet.js security headers (CSP, HSTS, X-Frame-Options)
- Implemented rate limiting (100 req/15min general, 5 req/15min auth)
- Configured Sentry error monitoring

## [1.0.0] - 2025-12-01

### Added
- **Initial Release**: TRUE ASI - Artificial Superintelligence System
- **Manus OAuth Integration**: Secure authentication with session cookies
- **Database Schema**: 11 tables with proper relationships
- **tRPC API**: Type-safe procedures for all backend operations
- **S3 Storage**: File upload and storage with AWS S3
- **LLM Integration**: `invokeLLM()` helper for AI model interactions
- **WebSocket Support**: Real-time updates via Socket.io

### Features
- User authentication and authorization
- Business analysis with AI recommendations
- Revenue tracking and metrics
- Execution dashboard for automation workflows
- S-7 test submissions and leaderboard

### Infrastructure
- Express 4 server with Vite frontend
- MySQL/TiDB database with Drizzle ORM
- React 19 with TypeScript
- Tailwind CSS 4 for styling
- shadcn/ui components

---

## Version History Summary

- **v1.3.0** (2025-12-05): Real-time WebSocket + Notification Center
- **v1.2.0** (2025-12-04): Premium UI + Multi-Model Consensus + CSV Export
- **v1.1.0** (2025-12-03): Revenue Tracking + Analysis History + Deeplink Registry
- **v1.0.0** (2025-12-01): Initial Release - TRUE ASI System

---

## Upgrade Notes

### Upgrading to v1.3.0
- No database migrations required (tables auto-created on first use)
- WebSocket connections now support authentication via session cookies
- Notification center requires tRPC client regeneration: `pnpm build`

### Upgrading to v1.2.0
- Run `pnpm install` to update dependencies
- No breaking changes to existing APIs

### Upgrading to v1.1.0
- Database schema changes require migration: `pnpm db:push`
- Update environment variables if using custom API keys

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

Proprietary - All Rights Reserved
