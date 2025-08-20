"""
Session management for Debabelizer

Tracks active streaming sessions, provider usage, and resource cleanup.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about an active session"""
    session_id: str
    session_type: str  # "stt" or "tts"
    provider: str
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]


class SessionManager:
    """Manages active sessions and resource cleanup"""
    
    def __init__(self, cleanup_interval: int = 300):  # 5 minutes
        """
        Initialize session manager
        
        Args:
            cleanup_interval: Seconds between cleanup runs
        """
        self.sessions: Dict[str, SessionInfo] = {}
        self.cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
    def create_session(
        self,
        session_id: str,
        session_type: str,
        provider: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionInfo:
        """
        Create a new session
        
        Args:
            session_id: Unique session identifier
            session_type: Type of session ("stt" or "tts")
            provider: Provider name
            metadata: Additional session metadata
            
        Returns:
            SessionInfo object
        """
        now = datetime.now()
        
        session_info = SessionInfo(
            session_id=session_id,
            session_type=session_type,
            provider=provider,
            created_at=now,
            last_activity=now,
            metadata=metadata or {}
        )
        
        self.sessions[session_id] = session_info
        
        # Start cleanup task if not running (will only start if event loop exists)
        if not self._running:
            self._start_cleanup_task()
            
        logger.debug(f"Created {session_type} session {session_id} with {provider}")
        return session_info
        
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information"""
        return self.sessions.get(session_id)
        
    def update_session_activity(self, session_id: str) -> None:
        """Update last activity timestamp for a session"""
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = datetime.now()
            
    def end_session(self, session_id: str) -> bool:
        """
        End a session
        
        Args:
            session_id: Session to end
            
        Returns:
            True if session was found and ended, False otherwise
        """
        if session_id in self.sessions:
            session_info = self.sessions.pop(session_id)
            duration = datetime.now() - session_info.created_at
            logger.debug(
                f"Ended {session_info.session_type} session {session_id} "
                f"({duration.total_seconds():.1f}s duration)"
            )
            return True
        return False
        
    def get_active_sessions(self) -> List[SessionInfo]:
        """Get list of all active sessions"""
        return list(self.sessions.values())
        
    def get_sessions_by_type(self, session_type: str) -> List[SessionInfo]:
        """Get sessions of a specific type"""
        return [
            session for session in self.sessions.values()
            if session.session_type == session_type
        ]
        
    def get_sessions_by_provider(self, provider: str) -> List[SessionInfo]:
        """Get sessions for a specific provider"""
        return [
            session for session in self.sessions.values()
            if session.provider == provider
        ]
        
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        if not self.sessions:
            return {
                "total_sessions": 0,
                "by_type": {},
                "by_provider": {},
                "oldest_session": None,
                "newest_session": None
            }
            
        sessions = list(self.sessions.values())
        
        # Count by type
        by_type = {}
        for session in sessions:
            by_type[session.session_type] = by_type.get(session.session_type, 0) + 1
            
        # Count by provider
        by_provider = {}
        for session in sessions:
            by_provider[session.provider] = by_provider.get(session.provider, 0) + 1
            
        # Find oldest and newest
        oldest = min(sessions, key=lambda s: s.created_at)
        newest = max(sessions, key=lambda s: s.created_at)
        
        return {
            "total_sessions": len(sessions),
            "by_type": by_type,
            "by_provider": by_provider,
            "oldest_session": {
                "id": oldest.session_id,
                "created_at": oldest.created_at,
                "age_seconds": (datetime.now() - oldest.created_at).total_seconds()
            },
            "newest_session": {
                "id": newest.session_id,
                "created_at": newest.created_at,
                "age_seconds": (datetime.now() - newest.created_at).total_seconds()
            }
        }
        
    def _start_cleanup_task(self):
        """Start the background cleanup task"""
        try:
            # Only start if there's an event loop running
            loop = asyncio.get_running_loop()
            if not self._cleanup_task or self._cleanup_task.done():
                self._running = True
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
                logger.debug("Started session cleanup task")
        except RuntimeError:
            # No event loop running, cleanup will be deferred
            logger.debug("No event loop for cleanup task, deferring until needed")
    
    async def start_cleanup_if_needed(self):
        """Manually start cleanup task if not running and event loop is available"""
        if not self._running:
            self._start_cleanup_task()
            
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_stale_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup loop: {e}")
                
    async def _cleanup_stale_sessions(self):
        """Clean up stale sessions"""
        now = datetime.now()
        stale_threshold = timedelta(minutes=30)  # 30 minutes without activity
        
        stale_sessions = []
        for session_id, session_info in self.sessions.items():
            if now - session_info.last_activity > stale_threshold:
                stale_sessions.append(session_id)
                
        for session_id in stale_sessions:
            self.end_session(session_id)
            logger.info(f"Cleaned up stale session: {session_id}")
            
        if stale_sessions:
            logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")
            
    async def cleanup_all_sessions(self):
        """Clean up all sessions and stop cleanup task"""
        self._running = False
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        # End all sessions
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            self.end_session(session_id)
            
        logger.info(f"Cleaned up all {len(session_ids)} sessions")
        
    def __len__(self) -> int:
        """Number of active sessions"""
        return len(self.sessions)