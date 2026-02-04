"""
MCP Client implementation.
Provides interface to interact with MCP Server.
"""

from typing import Dict, Any, Optional
import requests
from dataclasses import dataclass, asdict

from medassist.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MCPClientRequest:
    """Client request to MCP Server"""
    tool: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


@dataclass
class MCPClientResponse:
    """Response from MCP Server"""
    success: bool
    result: Any
    metadata: Dict[str, Any]
    error: Optional[str] = None


class MCPClient:
    """
    MCP Client for interacting with MCP Server.
    Can be used locally or via HTTP API.
    """
    
    def __init__(
        self,
        server=None,
        api_url: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize MCP Client.
        
        Args:
            server: Local MCPServer instance
            api_url: Remote API URL (if using HTTP)
            timeout: Request timeout in seconds
        """
        self.server = server
        self.api_url = api_url
        self.timeout = timeout
        
        if not server and not api_url:
            raise ValueError("Must provide either server instance or api_url")
        
        logger.info("MCP Client initialized")
    
    def send_request(
        self,
        tool: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> MCPClientResponse:
        """
        Send request to MCP Server.
        
        Args:
            tool: Tool or skill name
            parameters: Tool parameters
            context: Additional context
            
        Returns:
            MCPClientResponse with result
        """
        request = MCPClientRequest(
            tool=tool,
            parameters=parameters,
            context=context
        )
        
        if self.server:
            # Local server
            return self._send_local(request)
        else:
            # Remote API
            return self._send_remote(request)
    
    def _send_local(self, request: MCPClientRequest) -> MCPClientResponse:
        """Send request to local server"""
        try:
            from medassist.mcp_server import MCPRequest
            
            mcp_request = MCPRequest(
                tool=request.tool,
                parameters=request.parameters,
                context=request.context
            )
            
            response = self.server.process_request(mcp_request)
            
            return MCPClientResponse(
                success=response.success,
                result=response.result,
                metadata=response.metadata,
                error=response.error
            )
        except Exception as e:
            logger.error(f"Local request failed: {e}")
            return MCPClientResponse(
                success=False,
                result=None,
                metadata={},
                error=str(e)
            )
    
    def _send_remote(self, request: MCPClientRequest) -> MCPClientResponse:
        """Send request to remote API"""
        try:
            response = requests.post(
                f"{self.api_url}/mcp",
                json=asdict(request),
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return MCPClientResponse(**data)
            else:
                return MCPClientResponse(
                    success=False,
                    result=None,
                    metadata={},
                    error=f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"Remote request failed: {e}")
            return MCPClientResponse(
                success=False,
                result=None,
                metadata={},
                error=str(e)
            )
    
    # Convenience methods for common operations
    
    def kg_search(
        self,
        query: str,
        max_depth: int = 3,
        max_width: int = 10,
        method: str = "bfs"
    ) -> MCPClientResponse:
        """Search knowledge graph"""
        return self.send_request(
            tool="kg_search",
            parameters={
                "query": query,
                "max_depth": max_depth,
                "max_width": max_width,
                "method": method
            }
        )
    
    def kg_write(
        self,
        entities: list,
        relationships: list
    ) -> MCPClientResponse:
        """Write to knowledge graph"""
        return self.send_request(
            tool="kg_write",
            parameters={
                "entities": entities,
                "relationships": relationships
            }
        )
    
    def web_search(
        self,
        query: str,
        max_results: int = 5
    ) -> MCPClientResponse:
        """Search web (PubMed)"""
        return self.send_request(
            tool="web_search",
            parameters={
                "query": query,
                "max_results": max_results
            }
        )
    
    def update_memory(self, content: str) -> MCPClientResponse:
        """Update knowledge graph memory"""
        return self.send_request(
            tool="update_memory",
            parameters={"content": content}
        )
    
    def write_content(
        self,
        topic: str,
        include_evidence: bool = True
    ) -> MCPClientResponse:
        """Generate content about topic"""
        return self.send_request(
            tool="write_content",
            parameters={
                "topic": topic,
                "include_evidence": include_evidence
            }
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities"""
        if self.server:
            return self.server.get_capabilities()
        else:
            try:
                response = requests.get(
                    f"{self.api_url}/mcp/capabilities",
                    timeout=self.timeout
                )
                return response.json() if response.status_code == 200 else {}
            except Exception as e:
                logger.error(f"Failed to get capabilities: {e}")
                return {}


class InteractiveMCPClient:
    """
    Interactive MCP Client with user-friendly interface.
    Provides high-level methods for common workflows.
    """
    
    def __init__(self, client: MCPClient):
        """Initialize with MCPClient"""
        self.client = client
        logger.info("Interactive MCP Client initialized")
    
    def query_medical_topic(
        self,
        question: str,
        include_evidence: bool = True
    ) -> Dict[str, Any]:
        """
        Query a medical topic with full workflow.
        
        Combines KG search and evidence retrieval.
        """
        logger.info(f"Querying medical topic: {question}")
        
        # Search knowledge graph
        kg_response = self.client.kg_search(question)
        
        # Get evidence if requested
        evidence_response = None
        if include_evidence:
            evidence_response = self.client.web_search(question, max_results=3)
        
        return {
            "question": question,
            "knowledge_graph": kg_response.result if kg_response.success else [],
            "evidence": evidence_response.result if evidence_response and evidence_response.success else [],
            "kg_success": kg_response.success,
            "evidence_success": evidence_response.success if evidence_response else False
        }
    
    def learn_from_document(self, document: str, document_id: str = None) -> Dict[str, Any]:
        """
        Learn from a medical document.
        
        Extracts knowledge and updates memory.
        """
        logger.info(f"Learning from document: {document_id or 'unnamed'}")
        
        # Update memory with document content
        response = self.client.update_memory(document)
        
        return {
            "document_id": document_id,
            "success": response.success,
            "entities_extracted": response.result.get("entities_extracted", 0) if response.success else 0,
            "error": response.error
        }
    
    def generate_medical_report(
        self,
        topic: str,
        include_evidence: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a medical report on a topic.
        
        Uses write_content skill.
        """
        logger.info(f"Generating report on: {topic}")
        
        response = self.client.write_content(topic, include_evidence)
        
        if response.success:
            return {
                "topic": topic,
                "report": response.result,
                "success": True
            }
        else:
            return {
                "topic": topic,
                "success": False,
                "error": response.error
            }
    
    def explore_connections(
        self,
        entity: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Explore connections in knowledge graph.
        
        Shows multi-hop relationships.
        """
        logger.info(f"Exploring connections for: {entity}")
        
        response = self.client.kg_search(
            entity,
            max_depth=depth,
            max_width=15,
            method="bfs"
        )
        
        return {
            "entity": entity,
            "depth": depth,
            "connections": response.result if response.success else [],
            "success": response.success
        }
