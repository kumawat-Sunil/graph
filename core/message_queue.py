"""
Message Queue system for asynchronous agent communication.

This module provides a lightweight message queue for routing messages
between agents in the Graph-Enhanced Agentic RAG system.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, List, Callable
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from .interfaces import MessageType
from .protocols import AgentMessage

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class QueuedMessage:
    """A message in the queue with metadata."""
    message: AgentMessage
    priority: MessagePriority = MessagePriority.NORMAL
    queued_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 30
    
    def is_expired(self) -> bool:
        """Check if the message has expired."""
        return datetime.now() > self.queued_at + timedelta(seconds=self.timeout_seconds)
    
    def can_retry(self) -> bool:
        """Check if the message can be retried."""
        return self.retry_count < self.max_retries


class MessageQueue:
    """
    Asynchronous message queue for agent communication.
    
    Features:
    - Priority-based message ordering
    - Message retry logic with exponential backoff
    - Dead letter queue for failed messages
    - Message expiration and cleanup
    - Metrics and monitoring
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.queues: Dict[MessagePriority, deque] = {
            priority: deque() for priority in MessagePriority
        }
        self.dead_letter_queue: deque = deque(maxlen=100)
        self.processing_messages: Dict[str, QueuedMessage] = {}
        
        # Statistics
        self.stats = {
            "messages_queued": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "messages_expired": 0,
            "messages_retried": 0
        }
        
        # Event handlers
        self.message_handlers: Dict[str, Callable] = {}
        
        # Queue processing
        self._processing = False
        self._processor_task: Optional[asyncio.Task] = None
        
        logger.info("Message Queue initialized")
    
    async def start_processing(self):
        """Start the message processing loop."""
        if self._processing:
            logger.warning("Message queue processing already started")
            return
        
        self._processing = True
        self._processor_task = asyncio.create_task(self._process_messages())
        logger.info("✅ Message queue processing started")
    
    async def stop_processing(self):
        """Stop the message processing loop."""
        if not self._processing:
            return
        
        self._processing = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Message queue processing stopped")
    
    def enqueue(self, message: AgentMessage, priority: MessagePriority = MessagePriority.NORMAL, 
                timeout_seconds: int = 30, max_retries: int = 3) -> bool:
        """
        Add a message to the queue.
        
        Args:
            message: The message to queue
            priority: Message priority level
            timeout_seconds: Message timeout in seconds
            max_retries: Maximum retry attempts
            
        Returns:
            bool: True if message was queued successfully
        """
        try:
            # Check queue size limits
            total_queued = sum(len(queue) for queue in self.queues.values())
            if total_queued >= self.max_queue_size:
                logger.error(f"❌ Queue full, dropping message: {message.id}")
                return False
            
            # Create queued message
            queued_msg = QueuedMessage(
                message=message,
                priority=priority,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries
            )
            
            # Add to appropriate priority queue
            self.queues[priority].append(queued_msg)
            self.stats["messages_queued"] += 1
            
            logger.debug(f"📤 Queued message {message.id} with priority {priority.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error queuing message: {e}")
            return False
    
    def dequeue(self) -> Optional[QueuedMessage]:
        """
        Get the next message from the queue (highest priority first).
        
        Returns:
            QueuedMessage or None if queue is empty
        """
        # Process queues in priority order (highest first)
        for priority in sorted(MessagePriority, key=lambda p: p.value, reverse=True):
            queue = self.queues[priority]
            if queue:
                queued_msg = queue.popleft()
                
                # Check if message has expired
                if queued_msg.is_expired():
                    logger.warning(f"⏰ Message {queued_msg.message.id} expired")
                    self.stats["messages_expired"] += 1
                    self.dead_letter_queue.append(queued_msg)
                    continue
                
                # Add to processing messages
                self.processing_messages[queued_msg.message.id] = queued_msg
                return queued_msg
        
        return None
    
    def register_handler(self, agent_id: str, handler: Callable):
        """
        Register a message handler for an agent.
        
        Args:
            agent_id: ID of the agent
            handler: Async function to handle messages
        """
        self.message_handlers[agent_id] = handler
        logger.debug(f"📝 Registered message handler for {agent_id}")
    
    def unregister_handler(self, agent_id: str):
        """
        Unregister a message handler.
        
        Args:
            agent_id: ID of the agent
        """
        if agent_id in self.message_handlers:
            del self.message_handlers[agent_id]
            logger.debug(f"🗑️ Unregistered message handler for {agent_id}")
    
    async def _process_messages(self):
        """Main message processing loop."""
        logger.info("🔄 Starting message processing loop")
        
        while self._processing:
            try:
                # Get next message
                queued_msg = self.dequeue()
                
                if queued_msg is None:
                    # No messages, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # Process the message
                await self._process_single_message(queued_msg)
                
            except asyncio.CancelledError:
                logger.info("🛑 Message processing cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in message processing loop: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _process_single_message(self, queued_msg: QueuedMessage):
        """
        Process a single message.
        
        Args:
            queued_msg: The queued message to process
        """
        message = queued_msg.message
        
        try:
            logger.debug(f"🔄 Processing message {message.id} -> {message.target_agent_id}")
            
            # Get message handler
            handler = self.message_handlers.get(message.target_agent_id)
            
            if not handler:
                logger.error(f"❌ No handler for agent {message.target_agent_id}")
                await self._handle_message_failure(queued_msg, "No handler found")
                return
            
            # Process message with timeout
            try:
                response = await asyncio.wait_for(
                    handler(message),
                    timeout=queued_msg.timeout_seconds
                )
                
                # Message processed successfully
                await self._handle_message_success(queued_msg, response)
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Message {message.id} timed out")
                await self._handle_message_failure(queued_msg, "Timeout")
                
        except Exception as e:
            logger.error(f"❌ Error processing message {message.id}: {e}")
            await self._handle_message_failure(queued_msg, str(e))
    
    async def _handle_message_success(self, queued_msg: QueuedMessage, response: Any):
        """Handle successful message processing."""
        message = queued_msg.message
        
        # Remove from processing messages
        if message.id in self.processing_messages:
            del self.processing_messages[message.id]
        
        self.stats["messages_processed"] += 1
        logger.debug(f"✅ Message {message.id} processed successfully")
        
        # Handle response if needed (could trigger callback or send to another agent)
        if response and hasattr(response, 'correlation_id') and response.correlation_id:
            # This is a response to another message, could route it back
            pass
    
    async def _handle_message_failure(self, queued_msg: QueuedMessage, error: str):
        """Handle failed message processing."""
        message = queued_msg.message
        
        # Remove from processing messages
        if message.id in self.processing_messages:
            del self.processing_messages[message.id]
        
        # Check if we can retry
        if queued_msg.can_retry():
            queued_msg.retry_count += 1
            self.stats["messages_retried"] += 1
            
            # Exponential backoff
            delay = min(2 ** queued_msg.retry_count, 30)  # Max 30 seconds
            
            logger.warning(f"🔄 Retrying message {message.id} in {delay}s (attempt {queued_msg.retry_count})")
            
            # Re-queue with delay
            await asyncio.sleep(delay)
            self.queues[queued_msg.priority].append(queued_msg)
        else:
            # Max retries reached, move to dead letter queue
            logger.error(f"💀 Message {message.id} failed permanently: {error}")
            self.stats["messages_failed"] += 1
            self.dead_letter_queue.append(queued_msg)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        queue_sizes = {
            priority.name.lower(): len(queue) 
            for priority, queue in self.queues.items()
        }
        
        return {
            "queue_sizes": queue_sizes,
            "total_queued": sum(queue_sizes.values()),
            "processing_count": len(self.processing_messages),
            "dead_letter_count": len(self.dead_letter_queue),
            "statistics": self.stats.copy(),
            "is_processing": self._processing
        }
    
    def get_dead_letter_messages(self) -> List[Dict[str, Any]]:
        """Get messages from the dead letter queue."""
        return [
            {
                "message_id": msg.message.id,
                "target_agent": msg.message.target_agent_id,
                "message_type": msg.message.message_type.value,
                "queued_at": msg.queued_at.isoformat(),
                "retry_count": msg.retry_count,
                "priority": msg.priority.name
            }
            for msg in self.dead_letter_queue
        ]
    
    async def clear_expired_messages(self):
        """Clean up expired messages from all queues."""
        expired_count = 0
        
        for priority, queue in self.queues.items():
            # Create new queue without expired messages
            new_queue = deque()
            
            while queue:
                msg = queue.popleft()
                if msg.is_expired():
                    expired_count += 1
                    self.dead_letter_queue.append(msg)
                else:
                    new_queue.append(msg)
            
            self.queues[priority] = new_queue
        
        if expired_count > 0:
            self.stats["messages_expired"] += expired_count
            logger.info(f"🧹 Cleaned up {expired_count} expired messages")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the message queue."""
        stats = self.get_queue_stats()
        
        # Determine health status
        total_queued = stats["total_queued"]
        processing_count = stats["processing_count"]
        dead_letter_count = stats["dead_letter_count"]
        
        status = "healthy"
        issues = []
        
        # Check for queue congestion
        if total_queued > self.max_queue_size * 0.8:
            status = "degraded"
            issues.append("Queue approaching capacity")
        
        # Check for too many processing messages
        if processing_count > 50:
            status = "degraded"
            issues.append("High number of processing messages")
        
        # Check for too many dead letters
        if dead_letter_count > 20:
            status = "degraded"
            issues.append("High number of failed messages")
        
        # Check if processing is stopped
        if not self._processing:
            status = "unhealthy"
            issues.append("Message processing is stopped")
        
        return {
            "status": status,
            "issues": issues,
            "statistics": stats,
            "uptime_seconds": (datetime.now() - datetime.now()).total_seconds()  # Placeholder
        }


# Global message queue instance
_message_queue: Optional[MessageQueue] = None


def get_message_queue() -> MessageQueue:
    """Get the global message queue instance."""
    global _message_queue
    
    if _message_queue is None:
        _message_queue = MessageQueue()
    
    return _message_queue


async def initialize_message_queue() -> MessageQueue:
    """Initialize the global message queue."""
    global _message_queue
    _message_queue = MessageQueue()
    await _message_queue.start_processing()
    logger.info("✅ Message Queue initialized and started")
    return _message_queue


async def shutdown_message_queue() -> None:
    """Shutdown the global message queue."""
    global _message_queue
    if _message_queue:
        await _message_queue.stop_processing()
        _message_queue = None
        logger.info("✅ Message Queue shutdown complete")