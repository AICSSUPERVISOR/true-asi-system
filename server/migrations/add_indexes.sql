-- Database Optimization: Add Indexes for Performance
-- This migration adds indexes to frequently queried columns

-- Users table indexes
CREATE INDEX IF NOT EXISTS idx_users_openId ON users(openId);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_createdAt ON users(createdAt);
CREATE INDEX IF NOT EXISTS idx_users_lastSignedIn ON users(lastSignedIn);

-- S7 Submissions indexes
CREATE INDEX IF NOT EXISTS idx_s7_submissions_userId ON s7_submissions(userId);
CREATE INDEX IF NOT EXISTS idx_s7_submissions_questionNumber ON s7_submissions(questionNumber);
CREATE INDEX IF NOT EXISTS idx_s7_submissions_submittedAt ON s7_submissions(submittedAt);
CREATE INDEX IF NOT EXISTS idx_s7_submissions_averageScore ON s7_submissions(averageScore);
CREATE INDEX IF NOT EXISTS idx_s7_submissions_meetsThreshold ON s7_submissions(meetsThreshold);

-- S7 Rankings indexes
CREATE INDEX IF NOT EXISTS idx_s7_rankings_userId ON s7_rankings(userId);
CREATE INDEX IF NOT EXISTS idx_s7_rankings_averageScore ON s7_rankings(averageScore);
CREATE INDEX IF NOT EXISTS idx_s7_rankings_bestScore ON s7_rankings(bestScore);
CREATE INDEX IF NOT EXISTS idx_s7_rankings_rank ON s7_rankings(rank);

-- Answer Comparisons indexes
CREATE INDEX IF NOT EXISTS idx_answer_comparisons_userId ON answer_comparisons(userId);
CREATE INDEX IF NOT EXISTS idx_answer_comparisons_questionNumber ON answer_comparisons(questionNumber);
CREATE INDEX IF NOT EXISTS idx_answer_comparisons_userSubmissionId ON answer_comparisons(userSubmissionId);
CREATE INDEX IF NOT EXISTS idx_answer_comparisons_createdAt ON answer_comparisons(createdAt);

-- Business Analyses indexes (if table exists)
CREATE INDEX IF NOT EXISTS idx_analyses_userId ON analyses(userId);
CREATE INDEX IF NOT EXISTS idx_analyses_createdAt ON analyses(createdAt);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);

-- Recommendations indexes (if table exists)
CREATE INDEX IF NOT EXISTS idx_recommendations_analysisId ON recommendations(analysisId);
CREATE INDEX IF NOT EXISTS idx_recommendations_priority ON recommendations(priority);
CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status);

-- Executions indexes (if table exists)
CREATE INDEX IF NOT EXISTS idx_executions_workflowId ON executions(workflowId);
CREATE INDEX IF NOT EXISTS idx_executions_status ON executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_startedAt ON executions(startedAt);

-- Revenue Tracking indexes (if table exists)
CREATE INDEX IF NOT EXISTS idx_revenue_tracking_userId ON revenue_tracking(userId);
CREATE INDEX IF NOT EXISTS idx_revenue_tracking_date ON revenue_tracking(date);
CREATE INDEX IF NOT EXISTS idx_revenue_tracking_createdAt ON revenue_tracking(createdAt);

-- Notifications indexes
CREATE INDEX IF NOT EXISTS idx_notifications_userId ON notifications(userId);
CREATE INDEX IF NOT EXISTS idx_notifications_isRead ON notifications(isRead);
CREATE INDEX IF NOT EXISTS idx_notifications_createdAt ON notifications(createdAt);
CREATE INDEX IF NOT EXISTS idx_notifications_type ON notifications(type);

-- Scheduled Exports indexes
CREATE INDEX IF NOT EXISTS idx_scheduled_exports_userId ON scheduled_exports(userId);
CREATE INDEX IF NOT EXISTS idx_scheduled_exports_isActive ON scheduled_exports(isActive);
CREATE INDEX IF NOT EXISTS idx_scheduled_exports_nextRunAt ON scheduled_exports(nextRunAt);

-- Export History indexes
CREATE INDEX IF NOT EXISTS idx_export_history_scheduleId ON export_history(scheduleId);
CREATE INDEX IF NOT EXISTS idx_export_history_userId ON export_history(userId);
CREATE INDEX IF NOT EXISTS idx_export_history_executedAt ON export_history(executedAt);
CREATE INDEX IF NOT EXISTS idx_export_history_status ON export_history(status);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_s7_submissions_user_question ON s7_submissions(userId, questionNumber);
CREATE INDEX IF NOT EXISTS idx_notifications_user_unread ON notifications(userId, isRead);
CREATE INDEX IF NOT EXISTS idx_revenue_tracking_user_date ON revenue_tracking(userId, date);
