CREATE TABLE `export_history` (
	`id` varchar(128) NOT NULL,
	`userId` int NOT NULL,
	`scheduledExportId` varchar(128),
	`exportType` varchar(100) NOT NULL,
	`fileName` varchar(255) NOT NULL,
	`fileSize` int,
	`fileUrl` varchar(512),
	`status` enum('pending','completed','failed') NOT NULL DEFAULT 'pending',
	`emailSent` int NOT NULL DEFAULT 0,
	`emailRecipients` text,
	`errorMessage` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`completedAt` timestamp,
	CONSTRAINT `export_history_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `notifications` (
	`id` varchar(128) NOT NULL,
	`userId` int NOT NULL,
	`title` varchar(255) NOT NULL,
	`message` text NOT NULL,
	`type` enum('info','success','warning','error','analysis_complete','execution_complete') NOT NULL DEFAULT 'info',
	`analysisId` varchar(128),
	`workflowId` varchar(128),
	`link` varchar(512),
	`isRead` int NOT NULL DEFAULT 0,
	`readAt` timestamp,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `notifications_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `scheduled_exports` (
	`id` varchar(128) NOT NULL,
	`userId` int NOT NULL,
	`name` varchar(255) NOT NULL,
	`exportType` enum('revenue_tracking','analysis_history','execution_history') NOT NULL,
	`analysisId` varchar(128),
	`frequency` enum('daily','weekly','monthly') NOT NULL,
	`dayOfWeek` int,
	`dayOfMonth` int,
	`timeOfDay` varchar(5) NOT NULL,
	`timezone` varchar(64) NOT NULL DEFAULT 'UTC',
	`emailRecipients` text NOT NULL,
	`includeCharts` int NOT NULL DEFAULT 0,
	`isActive` int NOT NULL DEFAULT 1,
	`lastRunAt` timestamp,
	`nextRunAt` timestamp,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `scheduled_exports_id` PRIMARY KEY(`id`)
);
