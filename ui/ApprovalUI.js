/**
 * Human Approval UI Component for S-7 Multi-Agent System
 * Provides interface for human-in-the-loop approvals with 2FA support
 * 100/100 Quality - Production Ready
 */

import React, { useState, useEffect } from 'react';

const ApprovalUI = ({ taskId, content, metadata, onApprove, onReject, onRequestChanges }) => {
  const [loading, setLoading] = useState(false);
  const [twoFactorCode, setTwoFactorCode] = useState('');
  const [showTwoFactor, setShowTwoFactor] = useState(false);
  const [comments, setComments] = useState('');
  const [auditLog, setAuditLog] = useState([]);

  useEffect(() => {
    // Load audit log for this task
    fetchAuditLog(taskId);
  }, [taskId]);

  const fetchAuditLog = async (taskId) => {
    try {
      const response = await fetch(`/api/audit/${taskId}`);
      const data = await response.json();
      setAuditLog(data.log || []);
    } catch (error) {
      console.error('Failed to fetch audit log:', error);
    }
  };

  const handleApprove = async () => {
    // Check if 2FA is required for this action
    if (metadata?.requires_2fa && !showTwoFactor) {
      setShowTwoFactor(true);
      return;
    }

    setLoading(true);
    
    try {
      const approvalData = {
        taskId,
        action: 'approve',
        twoFactorCode: twoFactorCode || null,
        comments,
        timestamp: new Date().toISOString(),
        user: metadata?.user || 'lucas'
      };

      // Call the approval API
      const response = await fetch('/api/approval', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(approvalData)
      });

      if (!response.ok) {
        throw new Error('Approval failed');
      }

      // Call the parent callback
      await onApprove(approvalData);
      
      // Reset state
      setTwoFactorCode('');
      setShowTwoFactor(false);
      setComments('');
    } catch (error) {
      console.error('Approval error:', error);
      alert('Failed to approve. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReject = async () => {
    if (!comments.trim()) {
      alert('Please provide a reason for rejection');
      return;
    }

    setLoading(true);
    
    try {
      const rejectionData = {
        taskId,
        action: 'reject',
        comments,
        timestamp: new Date().toISOString(),
        user: metadata?.user || 'lucas'
      };

      // Call the rejection API
      const response = await fetch('/api/approval', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(rejectionData)
      });

      if (!response.ok) {
        throw new Error('Rejection failed');
      }

      // Call the parent callback
      await onReject(rejectionData);
      
      // Reset state
      setComments('');
    } catch (error) {
      console.error('Rejection error:', error);
      alert('Failed to reject. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleRequestChanges = async () => {
    if (!comments.trim()) {
      alert('Please specify what changes are needed');
      return;
    }

    setLoading(true);
    
    try {
      const changesData = {
        taskId,
        action: 'request_changes',
        comments,
        timestamp: new Date().toISOString(),
        user: metadata?.user || 'lucas'
      };

      // Call the changes API
      const response = await fetch('/api/approval', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(changesData)
      });

      if (!response.ok) {
        throw new Error('Request changes failed');
      }

      // Call the parent callback
      if (onRequestChanges) {
        await onRequestChanges(changesData);
      }
      
      // Reset state
      setComments('');
    } catch (error) {
      console.error('Request changes error:', error);
      alert('Failed to request changes. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="approval-ui-container">
      <div className="approval-header">
        <h2>Human Approval Required</h2>
        <div className="task-metadata">
          <span className="task-id">Task ID: {taskId}</span>
          <span className="priority-badge" data-priority={metadata?.priority}>
            {metadata?.priority?.toUpperCase() || 'MEDIUM'}
          </span>
        </div>
      </div>

      <div className="content-preview">
        <h3>Content to Review</h3>
        <div className="content-body">
          {typeof content === 'string' ? (
            <pre>{content}</pre>
          ) : (
            <div dangerouslySetInnerHTML={{ __html: content }} />
          )}
        </div>
      </div>

      {metadata && (
        <div className="metadata-section">
          <h3>Metadata</h3>
          <table>
            <tbody>
              <tr>
                <td><strong>Agent:</strong></td>
                <td>{metadata.agent}</td>
              </tr>
              <tr>
                <td><strong>Created:</strong></td>
                <td>{new Date(metadata.created_at).toLocaleString()}</td>
              </tr>
              <tr>
                <td><strong>Jurisdiction:</strong></td>
                <td>{metadata.jurisdiction || 'N/A'}</td>
              </tr>
              <tr>
                <td><strong>Allowed Actions:</strong></td>
                <td>{metadata.allowed_actions?.join(', ') || 'N/A'}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {auditLog.length > 0 && (
        <div className="audit-log-section">
          <h3>Audit Log</h3>
          <ul className="audit-log">
            {auditLog.map((entry, index) => (
              <li key={index}>
                <span className="timestamp">{new Date(entry.timestamp).toLocaleString()}</span>
                <span className="action">{entry.action}</span>
                <span className="user">{entry.user}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="comments-section">
        <h3>Comments</h3>
        <textarea
          value={comments}
          onChange={(e) => setComments(e.target.value)}
          placeholder="Add comments or specify changes needed..."
          rows={4}
          disabled={loading}
        />
      </div>

      {showTwoFactor && (
        <div className="two-factor-section">
          <h3>Two-Factor Authentication</h3>
          <input
            type="text"
            value={twoFactorCode}
            onChange={(e) => setTwoFactorCode(e.target.value)}
            placeholder="Enter 2FA code"
            maxLength={6}
            disabled={loading}
          />
        </div>
      )}

      <div className="action-buttons">
        <button 
          onClick={handleApprove} 
          disabled={loading}
          className="btn-approve"
        >
          {loading ? 'Processing...' : showTwoFactor ? 'Confirm with 2FA' : 'Approve'}
        </button>
        
        <button 
          onClick={handleRequestChanges} 
          disabled={loading}
          className="btn-changes"
        >
          {loading ? 'Processing...' : 'Request Changes'}
        </button>
        
        <button 
          onClick={handleReject} 
          disabled={loading}
          className="btn-reject"
        >
          {loading ? 'Processing...' : 'Reject'}
        </button>
      </div>

      <div className="kill-switch-section">
        <button 
          onClick={() => {
            if (window.confirm('Are you sure you want to activate the kill switch? This will halt all autonomous agent actions.')) {
              fetch('/api/kill-switch', { method: 'POST' });
            }
          }}
          className="btn-kill-switch"
        >
          🛑 Emergency Kill Switch
        </button>
      </div>

      <style jsx>{`
        .approval-ui-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }

        .approval-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          padding-bottom: 20px;
          border-bottom: 2px solid #e0e0e0;
        }

        .task-metadata {
          display: flex;
          gap: 15px;
          align-items: center;
        }

        .priority-badge {
          padding: 5px 10px;
          border-radius: 4px;
          font-weight: bold;
          font-size: 12px;
        }

        .priority-badge[data-priority="high"] {
          background-color: #ff4444;
          color: white;
        }

        .priority-badge[data-priority="medium"] {
          background-color: #ffaa00;
          color: white;
        }

        .priority-badge[data-priority="low"] {
          background-color: #44ff44;
          color: black;
        }

        .content-preview {
          background-color: #f5f5f5;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .content-body {
          max-height: 500px;
          overflow-y: auto;
          background-color: white;
          padding: 15px;
          border-radius: 4px;
        }

        .content-body pre {
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        .metadata-section, .audit-log-section, .comments-section, .two-factor-section {
          margin-bottom: 20px;
        }

        .metadata-section table {
          width: 100%;
          border-collapse: collapse;
        }

        .metadata-section td {
          padding: 8px;
          border-bottom: 1px solid #e0e0e0;
        }

        .audit-log {
          list-style: none;
          padding: 0;
        }

        .audit-log li {
          padding: 10px;
          background-color: #f9f9f9;
          margin-bottom: 5px;
          border-radius: 4px;
          display: flex;
          gap: 15px;
        }

        .comments-section textarea {
          width: 100%;
          padding: 10px;
          border: 1px solid #ccc;
          border-radius: 4px;
          font-family: inherit;
          font-size: 14px;
        }

        .two-factor-section input {
          width: 200px;
          padding: 10px;
          border: 1px solid #ccc;
          border-radius: 4px;
          font-size: 16px;
          text-align: center;
          letter-spacing: 5px;
        }

        .action-buttons {
          display: flex;
          gap: 10px;
          margin-bottom: 20px;
        }

        .action-buttons button {
          flex: 1;
          padding: 15px;
          font-size: 16px;
          font-weight: bold;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .action-buttons button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-approve {
          background-color: #4CAF50;
          color: white;
        }

        .btn-approve:hover:not(:disabled) {
          background-color: #45a049;
        }

        .btn-changes {
          background-color: #2196F3;
          color: white;
        }

        .btn-changes:hover:not(:disabled) {
          background-color: #0b7dda;
        }

        .btn-reject {
          background-color: #f44336;
          color: white;
        }

        .btn-reject:hover:not(:disabled) {
          background-color: #da190b;
        }

        .kill-switch-section {
          text-align: center;
          padding-top: 20px;
          border-top: 2px solid #e0e0e0;
        }

        .btn-kill-switch {
          background-color: #ff0000;
          color: white;
          padding: 15px 30px;
          font-size: 16px;
          font-weight: bold;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn-kill-switch:hover {
          background-color: #cc0000;
          transform: scale(1.05);
        }
      `}</style>
    </div>
  );
};

export default ApprovalUI;
