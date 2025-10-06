"""
Agent Registry for managing agent instances and communication.

This module provides centralized agent management and message routing
for the Graph-Enhanced Agentic RAG system.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime
import uuid
from dataclasses import dataclass, field

from .interfaces import MessageType
from .protocols import AgentMessage
from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    agent_instance: Any
    agent_type: str
    status: str = "active"
    registered_at: datetime = field(default_factory=datetime.now)
    last_activity: Optional[datetime] = None
    message_count: int = 0
    error_count: int = 0


class AgentRegistry:
    """
    Central registry for managing agent instances and routing messages.
    
    Provides:
    - Agent registration and discovery
    - Message routing between agents
    - Health monitoring and status tracking
    - Load balancing for multiple instances
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentInfo] = {}
        self.message_handlers: Dict[str, Any] = {}
        self.config = get_config()
        self._lock = asyncio.Lock()
        
        logger.info("Agent Registry initialized")
    
    async def register_agent(self, agent_id: str, agent_instance: Any, agent_type: str) -> bool:
        """
        Register an agent instance with the registry.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_instance: The actual agent instance
            agent_type: Type of agent (coordinator, graph_navigator, etc.)
            
        Returns:
            bool: True if registration successful
        """
        async with self._lock:
            try:
                if agent_id in self.agents:
                    logger.warning(f"Agent {agent_id} already registered, updating...")
                
                agent_info = AgentInfo(
                    agent_id=agent_id,
                    agent_instance=agent_instance,
                    agent_type=agent_type,
                    status="active"
                )
                
                self.agents[agent_id] = agent_info
                logger.info(f"✅ Registered agent: {agent_id} ({agent_type})")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to register agent {agent_id}: {e}")
                return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        async with self._lock:
            try:
                if agent_id in self.agents:
                    del self.agents[agent_id]
                    logger.info(f"✅ Unregistered agent: {agent_id}")
                    return True
                else:
                    logger.warning(f"⚠️ Agent {agent_id} not found for unregistration")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Failed to unregister agent {agent_id}: {e}")
                return False
    
    def get_agent(self, agent_id: str) -> Optional[Any]:
        """
        Get an agent instance by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            Agent instance or None if not found
        """
        agent_info = self.agents.get(agent_id)
        if agent_info and agent_info.status == "active":
            return agent_info.agent_instance
        return None
    
    def is_agent_available(self, agent_id: str) -> bool:
        """
        Check if an agent is available for processing.
        
        Args:
            agent_id: ID of the agent to check
            
        Returns:
            bool: True if agent is available
        """
        agent_info = self.agents.get(agent_id)
        return agent_info is not None and agent_info.status == "active"
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents with their status.
        
        Returns:
            List of agent information dictionaries
        """
        return [
            {
                "agent_id": info.agent_id,
                "agent_type": info.agent_type,
                "status": info.status,
                "registered_at": info.registered_at.isoformat(),
                "last_activity": info.last_activity.isoformat() if info.last_activity else None,
                "message_count": info.message_count,
                "error_count": info.error_count
            }
            for info in self.agents.values()
        ]
    
    async def send_message(self, target_agent_id: str, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Send a message to a specific agent.
        
        Args:
            target_agent_id: ID of the target agent
            message: Message to send
            
        Returns:
            Response message from the agent, if any
        """
        try:
            # Get target agent
            target_agent = self.get_agent(target_agent_id)
            if not target_agent:
                logger.error(f"❌ Target agent {target_agent_id} not found or unavailable")
                return None
            
            # Update message routing info
            message.target_agent_id = target_agent_id
            message.timestamp = datetime.now()
            
            # Send message to agent
            logger.debug(f"📤 Sending message to {target_agent_id}: {message.message_type}")
            
            # Update agent activity
            agent_info = self.agents.get(target_agent_id)
            if agent_info:
                agent_info.last_activity = datetime.now()
                agent_info.message_count += 1
            
            # Process message
            if hasattr(target_agent, 'process_message'):
                response = await target_agent.process_message(message)
                logger.debug(f"📥 Received response from {target_agent_id}")
                return response
            else:
                logger.error(f"❌ Agent {target_agent_id} does not support message processing")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error sending message to {target_agent_id}: {e}")
            
            # Update error count
            agent_info = self.agents.get(target_agent_id)
            if agent_info:
                agent_info.error_count += 1
            
            return None
    
    async def broadcast_message(self, message: AgentMessage, agent_types: Optional[List[str]] = None) -> Dict[str, Optional[AgentMessage]]:
        """
        Broadcast a message to multiple agents.
        
        Args:
            message: Message to broadcast
            agent_types: Optional list of agent types to target
            
        Returns:
            Dictionary mapping agent IDs to their responses
        """
        responses = {}
        
        # Filter agents by type if specified
        target_agents = []
        for agent_id, agent_info in self.agents.items():
            if agent_types is None or agent_info.agent_type in agent_types:
                if agent_info.status == "active":
                    target_agents.append(agent_id)
        
        # Send messages concurrently
        tasks = []
        for agent_id in target_agents:
            # Create a copy of the message for each agent
            agent_message = AgentMessage(
                agent_id=message.agent_id,
                target_agent_id=agent_id,
                message_type=message.message_type,
                payload=message.payload.copy() if isinstance(message.payload, dict) else message.payload,
                correlation_id=message.correlation_id
            )
            tasks.append(self.send_message(agent_id, agent_message))
        
        # Wait for all responses
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for agent_id, result in zip(target_agents, results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Error broadcasting to {agent_id}: {result}")
                    responses[agent_id] = None
                else:
                    responses[agent_id] = result
        
        return responses
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status information for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Status information dictionary or None
        """
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            return None
        
        return {
            "agent_id": agent_info.agent_id,
            "agent_type": agent_info.agent_type,
            "status": agent_info.status,
            "registered_at": agent_info.registered_at.isoformat(),
            "last_activity": agent_info.last_activity.isoformat() if agent_info.last_activity else None,
            "message_count": agent_info.message_count,
            "error_count": agent_info.error_count,
            "uptime_seconds": (datetime.now() - agent_info.registered_at).total_seconds()
        }
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get overall registry statistics.
        
        Returns:
            Registry statistics dictionary
        """
        total_agents = len(self.agents)
        active_agents = sum(1 for info in self.agents.values() if info.status == "active")
        total_messages = sum(info.message_count for info in self.agents.values())
        total_errors = sum(info.error_count for info in self.agents.values())
        
        agent_types = {}
        for info in self.agents.values():
            agent_types[info.agent_type] = agent_types.get(info.agent_type, 0) + 1
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "inactive_agents": total_agents - active_agents,
            "total_messages_processed": total_messages,
            "total_errors": total_errors,
            "agent_types": agent_types,
            "error_rate": total_errors / max(total_messages, 1)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all registered agents.
        
        Returns:
            Health check results
        """
        health_results = {
            "registry_status": "healthy",
            "total_agents": len(self.agents),
            "healthy_agents": 0,
            "unhealthy_agents": 0,
            "agent_health": {}
        }
        
        for agent_id, agent_info in self.agents.items():
            try:
                # Check if agent has health_check method
                if hasattr(agent_info.agent_instance, 'health_check'):
                    agent_health = await agent_info.agent_instance.health_check()
                    health_results["agent_health"][agent_id] = agent_health
                    
                    if agent_health.get("status") == "healthy":
                        health_results["healthy_agents"] += 1
                    else:
                        health_results["unhealthy_agents"] += 1
                else:
                    # Basic health check - agent is responsive
                    health_results["agent_health"][agent_id] = {
                        "status": "healthy" if agent_info.status == "active" else "unhealthy",
                        "last_activity": agent_info.last_activity.isoformat() if agent_info.last_activity else None
                    }
                    health_results["healthy_agents"] += 1
                    
            except Exception as e:
                logger.error(f"❌ Health check failed for agent {agent_id}: {e}")
                health_results["agent_health"][agent_id] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_results["unhealthy_agents"] += 1
        
        # Determine overall registry health
        if health_results["unhealthy_agents"] > 0:
            health_results["registry_status"] = "degraded"
        
        if health_results["healthy_agents"] == 0:
            health_results["registry_status"] = "unhealthy"
        
        return health_results


# Global agent registry instance
_agent_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _agent_registry
    
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    
    return _agent_registry


def initialize_agent_registry() -> AgentRegistry:
    """Initialize the global agent registry."""
    global _agent_registry
    _agent_registry = AgentRegistry()
    logger.info("✅ Agent Registry initialized")
    return _agent_registry


def shutdown_agent_registry() -> None:
    """Shutdown the global agent registry."""
    global _agent_registry
    if _agent_registry:
        # Unregister all agents
        for agent_id in list(_agent_registry.agents.keys()):
            asyncio.create_task(_agent_registry.unregister_agent(agent_id))
        
        _agent_registry = None
        logger.info("✅ Agent Registry shutdown complete")