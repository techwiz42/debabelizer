use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::Result;
use debabelizer_core::DebabelizerError;

#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    Active,
    Processing,
    Completed,
    Failed,
    Expired,
}

impl std::fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Processing => write!(f, "processing"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Expired => write!(f, "expired"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: Uuid,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub status: SessionStatus,
}

impl Session {
    pub fn new() -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Uuid::new_v4(),
            metadata: HashMap::new(),
            created_at: now,
            last_activity: now,
            status: SessionStatus::Active,
        }
    }
    
    pub fn with_id(id: Uuid) -> Self {
        let now = chrono::Utc::now();
        Self {
            id,
            metadata: HashMap::new(),
            created_at: now,
            last_activity: now,
            status: SessionStatus::Active,
        }
    }
    
    pub fn update_activity(&mut self) {
        self.last_activity = chrono::Utc::now();
    }
}

pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<Uuid, Session>>>,
    cleanup_handle: Option<JoinHandle<()>>,
}

impl SessionManager {
    pub fn new() -> Self {
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        
        // Start cleanup task if we're in an async context
        let cleanup_handle = if tokio::runtime::Handle::try_current().is_ok() {
            let sessions_clone = sessions.clone();
            Some(tokio::spawn(async move {
                let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300));
                loop {
                    interval.tick().await;
                    Self::cleanup_inactive_sessions(&sessions_clone).await;
                }
            }))
        } else {
            None
        };
        
        Self {
            sessions,
            cleanup_handle,
        }
    }
    
    pub async fn create_session(&self) -> Session {
        let session = Session::new();
        self.sessions.write().await.insert(session.id, session.clone());
        session
    }
    
    pub async fn get_session(&self, id: Uuid) -> Option<Session> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(&id) {
            session.update_activity();
            Some(session.clone())
        } else {
            None
        }
    }
    
    pub async fn update_session<F>(&self, id: Uuid, f: F) -> Result<()>
    where
        F: FnOnce(&mut Session),
    {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(&id) {
            f(session);
            session.update_activity();
            Ok(())
        } else {
            Err(DebabelizerError::Session(format!("Session {} not found", id)))
        }
    }
    
    pub async fn remove_session(&self, id: Uuid) -> Option<Session> {
        self.sessions.write().await.remove(&id)
    }
    
    pub async fn list_sessions(&self) -> Vec<Session> {
        self.sessions.read().await.values().cloned().collect()
    }
    
    async fn cleanup_inactive_sessions(sessions: &Arc<RwLock<HashMap<Uuid, Session>>>) {
        let now = chrono::Utc::now();
        let mut sessions = sessions.write().await;
        
        sessions.retain(|_, session| {
            let inactive_duration = now - session.last_activity;
            inactive_duration.num_hours() < 24
        });
    }
}

impl Drop for SessionManager {
    fn drop(&mut self) {
        if let Some(handle) = self.cleanup_handle.take() {
            handle.abort();
        }
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_session_lifecycle() {
        let manager = SessionManager::new();
        
        // Create session
        let session = manager.create_session().await;
        let session_id = session.id;
        
        // Get session
        let retrieved = manager.get_session(session_id).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, session_id);
        
        // Update session
        manager.update_session(session_id, |s| {
            s.metadata.insert("key".to_string(), serde_json::json!("value"));
        }).await.unwrap();
        
        let updated = manager.get_session(session_id).await.unwrap();
        assert_eq!(updated.metadata.get("key").unwrap(), "value");
        
        // Remove session
        let removed = manager.remove_session(session_id).await;
        assert!(removed.is_some());
        
        // Verify removed
        let not_found = manager.get_session(session_id).await;
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_session_creation() {
        let manager = SessionManager::new();
        
        let session = manager.create_session().await;
        
        // Test session properties
        assert!(!session.id.is_nil());
        assert_eq!(session.status, SessionStatus::Active);
        assert!(session.metadata.is_empty());
        
        // Test timestamps are recent
        let now = chrono::Utc::now();
        let time_diff = now - session.created_at;
        assert!(time_diff.num_seconds() < 5); // Created within 5 seconds
        
        let activity_diff = now - session.last_activity;
        assert!(activity_diff.num_seconds() < 5); // Activity within 5 seconds
    }

    #[tokio::test]
    async fn test_session_update_activity() {
        let manager = SessionManager::new();
        let session = manager.create_session().await;
        let session_id = session.id;
        let initial_activity = session.last_activity;
        
        // Wait a small amount
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        // Update session
        manager.update_session(session_id, |s| {
            s.metadata.insert("test".to_string(), serde_json::json!("data"));
        }).await.unwrap();
        
        let updated = manager.get_session(session_id).await.unwrap();
        assert!(updated.last_activity > initial_activity);
    }

    #[tokio::test]
    async fn test_session_update_nonexistent() {
        let manager = SessionManager::new();
        let fake_id = Uuid::new_v4();
        
        let result = manager.update_session(fake_id, |_| {}).await;
        assert!(result.is_err());
        
        if let Err(DebabelizerError::Session(msg)) = result {
            assert!(msg.contains("not found"));
        } else {
            panic!("Expected Session error");
        }
    }

    #[tokio::test]
    async fn test_multiple_sessions() {
        let manager = SessionManager::new();
        
        // Create multiple sessions
        let session1 = manager.create_session().await;
        let session2 = manager.create_session().await;
        let session3 = manager.create_session().await;
        
        assert_ne!(session1.id, session2.id);
        assert_ne!(session2.id, session3.id);
        assert_ne!(session1.id, session3.id);
        
        // Verify all sessions exist
        assert!(manager.get_session(session1.id).await.is_some());
        assert!(manager.get_session(session2.id).await.is_some());
        assert!(manager.get_session(session3.id).await.is_some());
        
        // List sessions
        let sessions = manager.list_sessions().await;
        assert_eq!(sessions.len(), 3);
        
        let session_ids: Vec<Uuid> = sessions.iter().map(|s| s.id).collect();
        assert!(session_ids.contains(&session1.id));
        assert!(session_ids.contains(&session2.id));
        assert!(session_ids.contains(&session3.id));
    }

    #[tokio::test]
    async fn test_session_metadata_operations() {
        let manager = SessionManager::new();
        let session = manager.create_session().await;
        let session_id = session.id;
        
        // Add multiple metadata entries
        manager.update_session(session_id, |s| {
            s.metadata.insert("user_id".to_string(), serde_json::json!("user123"));
            s.metadata.insert("language".to_string(), serde_json::json!("en-US"));
            s.metadata.insert("preferences".to_string(), serde_json::json!({
                "quality": "high",
                "streaming": true
            }));
        }).await.unwrap();
        
        let updated = manager.get_session(session_id).await.unwrap();
        assert_eq!(updated.metadata.get("user_id").unwrap(), "user123");
        assert_eq!(updated.metadata.get("language").unwrap(), "en-US");
        
        let prefs = updated.metadata.get("preferences").unwrap();
        assert_eq!(prefs["quality"], "high");
        assert_eq!(prefs["streaming"], true);
    }

    #[tokio::test]
    async fn test_session_status_transitions() {
        let manager = SessionManager::new();
        let session = manager.create_session().await;
        let session_id = session.id;
        
        // Initial status should be Active
        assert_eq!(session.status, SessionStatus::Active);
        
        // Update to Processing
        manager.update_session(session_id, |s| {
            s.status = SessionStatus::Processing;
        }).await.unwrap();
        
        let updated = manager.get_session(session_id).await.unwrap();
        assert_eq!(updated.status, SessionStatus::Processing);
        
        // Update to Completed
        manager.update_session(session_id, |s| {
            s.status = SessionStatus::Completed;
        }).await.unwrap();
        
        let completed = manager.get_session(session_id).await.unwrap();
        assert_eq!(completed.status, SessionStatus::Completed);
    }

    #[test]
    fn test_session_status_display() {
        assert_eq!(SessionStatus::Active.to_string(), "active");
        assert_eq!(SessionStatus::Processing.to_string(), "processing");
        assert_eq!(SessionStatus::Completed.to_string(), "completed");
        assert_eq!(SessionStatus::Failed.to_string(), "failed");
    }

    #[tokio::test]
    async fn test_session_removal() {
        let manager = SessionManager::new();
        
        // Create sessions
        let session1 = manager.create_session().await;
        let session2 = manager.create_session().await;
        let session1_id = session1.id;
        let session2_id = session2.id;
        
        // Remove one session
        let removed = manager.remove_session(session1_id).await;
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, session1_id);
        
        // Verify only second session remains
        assert!(manager.get_session(session1_id).await.is_none());
        assert!(manager.get_session(session2_id).await.is_some());
        
        let remaining_sessions = manager.list_sessions().await;
        assert_eq!(remaining_sessions.len(), 1);
        assert_eq!(remaining_sessions[0].id, session2_id);
    }

    #[tokio::test]
    async fn test_session_removal_nonexistent() {
        let manager = SessionManager::new();
        let fake_id = Uuid::new_v4();
        
        let result = manager.remove_session(fake_id).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_concurrent_session_operations() {
        let manager = Arc::new(SessionManager::new());
        let mut handles = vec![];
        
        // Create multiple concurrent operations
        for i in 0..10 {
            let manager_clone = manager.clone();
            let handle = tokio::spawn(async move {
                let session = manager_clone.create_session().await;
                let session_id = session.id;
                
                // Update session with unique data
                manager_clone.update_session(session_id, |s| {
                    s.metadata.insert("index".to_string(), serde_json::json!(i));
                }).await.unwrap();
                
                session_id
            });
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        let session_ids: Vec<Uuid> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|result| result.unwrap())
            .collect();
        
        // Verify all sessions were created
        assert_eq!(session_ids.len(), 10);
        
        let sessions = manager.list_sessions().await;
        assert_eq!(sessions.len(), 10);
        
        // Verify each session has unique data
        for (i, session_id) in session_ids.iter().enumerate() {
            let session = manager.get_session(*session_id).await.unwrap();
            assert_eq!(session.metadata.get("index").unwrap(), i);
        }
    }

    #[tokio::test]
    async fn test_session_manager_drop() {
        let manager = SessionManager::new();
        // In async context, cleanup_handle should be present
        assert!(manager.cleanup_handle.is_some());
        
        // Drop should not panic
        drop(manager);
    }

    #[tokio::test]
    async fn test_session_manager_default() {
        let manager = SessionManager::default();
        // In async context, cleanup_handle should be present
        assert!(manager.cleanup_handle.is_some());
    }

    #[tokio::test]
    async fn test_session_list_empty() {
        let manager = SessionManager::new();
        let sessions = manager.list_sessions().await;
        assert!(sessions.is_empty());
    }

    #[tokio::test]
    async fn test_session_with_error_status() {
        let manager = SessionManager::new();
        let session = manager.create_session().await;
        let session_id = session.id;
        
        // Set session to error status with error message
        manager.update_session(session_id, |s| {
            s.status = SessionStatus::Failed;
            s.metadata.insert("error".to_string(), serde_json::json!("Connection timeout"));
        }).await.unwrap();
        
        let error_session = manager.get_session(session_id).await.unwrap();
        assert_eq!(error_session.status, SessionStatus::Failed);
        assert_eq!(error_session.metadata.get("error").unwrap(), "Connection timeout");
    }
}