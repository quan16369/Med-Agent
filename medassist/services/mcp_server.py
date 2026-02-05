"""
Model Context Protocol (MCP) Server implementation.
Provides tools and skills for medical knowledge graph operations.
Enhanced with multimodal capabilities inspired by Kubrick AI.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from medassist.models.knowledge_graph import MedicalKnowledgeGraph
from medassist.tools.graph_retrieval import GraphConditionalRetrieval
from medassist.tools.pubmed_retrieval import PubMedRetriever
from medassist.tools.medical_ner import MedicalNER
from medassist.tools.multimodal import MultimodalProcessor, MultimodalInput, ImageProcessor
from medassist.models.multimodal_models import MultimodalMessage, ImageUrlContent, TextContent
from medassist.logging_utils import get_logger

logger = get_logger(__name__)


class ToolType(Enum):
    """Available MCP tool types"""
    KG_SEARCH = "kg_search"
    KG_WRITE = "kg_write"
    WEB_SEARCH = "web_search"
    GENERATE_IMAGE = "generate_image"
    LLM_TWIN = "llm_twin"


class SkillType(Enum):
    """Available MCP skill types"""
    UPDATE_MEMORY = "update_memory"
    WRITE_CONTENT = "write_content"


@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    tool_type: ToolType
    description: str
    parameters: Dict[str, Any]
    handler: Callable


@dataclass
class MCPSkill:
    """MCP Skill definition"""
    name: str
    skill_type: SkillType
    description: str
    tools_required: List[str]
    handler: Callable


@dataclass
class MCPRequest:
    """MCP Request from client"""
    tool: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


@dataclass
class MCPResponse:
    """MCP Response to client"""
    success: bool
    result: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class MCPServer:
    """
    Model Context Protocol Server.
    Orchestrates tools and skills for knowledge graph operations.
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[MedicalKnowledgeGraph] = None,
        graph_retriever: Optional[GraphConditionalRetrieval] = None,
        pubmed_retriever: Optional[PubMedRetriever] = None,
        ner: Optional[MedicalNER] = None,
        multimodal_processor: Optional[MultimodalProcessor] = None
    ):
        """Initialize MCP Server with components"""
        self.kg = knowledge_graph or MedicalKnowledgeGraph()
        self.graph_retriever = graph_retriever or GraphConditionalRetrieval(self.kg)
        self.pubmed = pubmed_retriever or PubMedRetriever()
        self.ner = ner or MedicalNER()
        self.multimodal = multimodal_processor or MultimodalProcessor()
        
        # Register tools and skills
        self.tools: Dict[str, MCPTool] = {}
        self.skills: Dict[str, MCPSkill] = {}
        self._register_tools()
        self._register_skills()
        
        logger.info("MCP Server initialized with tools and skills")
    
    def _register_tools(self):
        """Register available tools"""
        # KG Search tool
        self.tools["kg_search"] = MCPTool(
            name="kg_search",
            tool_type=ToolType.KG_SEARCH,
            description="Search medical knowledge graph for entities and relationships",
            parameters={
                "query": "str",
                "max_depth": "int",
                "max_width": "int",
                "method": "str (bfs/dfs)"
            },
            handler=self._handle_kg_search
        )
        
        # KG Write tool
        self.tools["kg_write"] = MCPTool(
            name="kg_write",
            tool_type=ToolType.KG_WRITE,
            description="Write entities and relationships to knowledge graph",
            parameters={
                "entities": "List[Dict]",
                "relationships": "List[Dict]"
            },
            handler=self._handle_kg_write
        )
        
        # Web Search tool (PubMed)
        self.tools["web_search"] = MCPTool(
            name="web_search",
            tool_type=ToolType.WEB_SEARCH,
            description="Search scientific literature on PubMed",
            parameters={
                "query": "str",
                "max_results": "int"
            },
            handler=self._handle_web_search
        )
        
        # Generate Image tool
        self.tools["generate_image"] = MCPTool(
            name="generate_image",
            tool_type=ToolType.GENERATE_IMAGE,
            description="Generate medical diagrams and illustrations",
            parameters={
                "description": "str",
                "diagram_type": "str (anatomy/flowchart/mechanism)"
            },
            handler=self._handle_generate_image
        )
        
        # Analyze Medical Image tool
        self.tools["analyze_image"] = MCPTool(
            name="analyze_image",
            tool_type=ToolType.GENERATE_IMAGE,
            description="Analyze medical images (X-ray, CT, MRI)",
            parameters={
                "image": "str (base64) or path",
                "image_type": "str (xray/ct/mri/pathology)"
            },
            handler=self._handle_analyze_image
        )
    
    def _register_skills(self):
        """Register available skills"""
        # Update Memory skill
        self.skills["update_memory"] = MCPSkill(
            name="update_memory",
            skill_type=SkillType.UPDATE_MEMORY,
            description="Update knowledge graph memory from new information",
            tools_required=["kg_write", "kg_search"],
            handler=self._handle_update_memory
        )
        
        # Write Content skill
        self.skills["write_content"] = MCPSkill(
            name="write_content",
            skill_type=SkillType.WRITE_CONTENT,
            description="Generate medical content from knowledge graph",
            tools_required=["kg_search", "web_search"],
            handler=self._handle_write_content
        )
    
    # Tool handlers
    
    def _handle_kg_search(self, parameters: Dict[str, Any]) -> MCPResponse:
        """Handle knowledge graph search"""
        try:
            query = parameters.get("query")
            max_depth = parameters.get("max_depth", 3)
            max_width = parameters.get("max_width", 10)
            method = parameters.get("method", "bfs")
            
            # Extract entities from query
            entities = self.ner.extract_entities(query)
            
            results = []
            for entity in entities:
                if method == "bfs":
                    paths = self.graph_retriever.bfs_retrieve(
                        entity.text, max_depth=max_depth, max_width=max_width
                    )
                else:
                    paths = self.graph_retriever.dfs_retrieve(
                        entity.text, max_depth=max_depth
                    )
                results.extend(paths)
            
            return MCPResponse(
                success=True,
                result=results,
                metadata={
                    "entities_found": len(entities),
                    "paths_found": len(results),
                    "method": method
                }
            )
        except Exception as e:
            logger.error(f"KG search failed: {e}")
            return MCPResponse(success=False, result=None, error=str(e))
    
    def _handle_kg_write(self, parameters: Dict[str, Any]) -> MCPResponse:
        """Handle knowledge graph write"""
        try:
            entities = parameters.get("entities", [])
            relationships = parameters.get("relationships", [])
            
            # Add entities
            for entity in entities:
                self.kg.add_entity(
                    entity.get("name"),
                    entity.get("type")
                )
            
            # Add relationships
            for rel in relationships:
                self.kg.add_relationship(
                    rel.get("source"),
                    rel.get("relation"),
                    rel.get("target")
                )
            
            return MCPResponse(
                success=True,
                result={
                    "entities_added": len(entities),
                    "relationships_added": len(relationships)
                },
                metadata={
                    "operation": "write",
                    "timestamp": "now"
                }
            )
        except Exception as e:
            logger.error(f"KG write failed: {e}")
            return MCPResponse(success=False, result=None, error=str(e))
    
    def _handle_web_search(self, parameters: Dict[str, Any]) -> MCPResponse:
        """Handle web search (PubMed)"""
        try:
            query = parameters.get("query")
            max_results = parameters.get("max_results", 5)
            
            results = self.pubmed.search(query, max_results=max_results)
            
            return MCPResponse(
                success=True,
                result=results,
                metadata={
                    "source": "pubmed",
                    "results_count": len(results)
                }
            )
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return MCPResponse(success=False, result=None, error=str(e))
    
    def _handle_generate_image(self, parameters: Dict[str, Any]) -> MCPResponse:
        """Handle medical diagram generation"""
        try:
            description = parameters.get("description")
            diagram_type = parameters.get("diagram_type", "anatomy")
            
            result = self.multimodal.generate_diagram(
                description,
                diagram_type
            )
            
            return MCPResponse(
                success=True,
                result=result,
                metadata={
                    "tool": "generate_image",
                    "diagram_type": diagram_type
                }
            )
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return MCPResponse(success=False, result=None, error=str(e))
    
    def _handle_analyze_image(self, parameters: Dict[str, Any]) -> MCPResponse:
        """Handle medical image analysis.
        
        Supports both base64 encoded images and file paths.
        Compatible with Kubrick-style image search patterns.
        """
        try:
            image = parameters.get("image")
            image_type = parameters.get("image_type", "xray")
            metadata = parameters.get("metadata", {})
            
            # Handle different input formats
            if isinstance(image, str):
                if image.startswith("data:"):
                    # Data URI format (Kubrick style)
                    image_processor = ImageProcessor()
                    pil_image = image_processor.decode_image_base64(image)
                    # Analyze PIL Image
                    result = self.multimodal.image_processor.analyze_medical_image(
                        image_processor.encode_image_base64(pil_image, "JPEG"),
                        image_type
                    )
                elif not image.startswith("/"):
                    # Plain base64
                    result = self.multimodal.image_processor.analyze_medical_image(
                        image,
                        image_type
                    )
                else:
                    # File path
                    image_processor = ImageProcessor()
                    pil_image = image_processor.load_image(image)
                    image_b64 = image_processor.encode_image_base64(pil_image, "JPEG")
                    result = self.multimodal.image_processor.analyze_medical_image(
                        image_b64,
                        image_type
                    )
            else:
                # Bytes
                result = self.multimodal.image_processor.analyze_medical_image(
                    image,
                    image_type
                )
            
            # Add metadata if provided
            if metadata:
                result["input_metadata"] = metadata
            
            return MCPResponse(
                success=True,
                result=result,
                metadata={
                    "tool": "analyze_image",
                    "image_type": image_type
                }
            )
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return MCPResponse(success=False, result=None, error=str(e))
    
    # Skill handlers
    
    def _handle_update_memory(self, parameters: Dict[str, Any]) -> MCPResponse:
        """Handle memory update skill"""
        try:
            content = parameters.get("content")
            
            # Extract entities from content
            entities = self.ner.extract_entities(content)
            
            # Prepare entities and relationships
            entity_list = [
                {"name": e.text, "type": e.label}
                for e in entities
            ]
            
            # Write to knowledge graph
            write_response = self._handle_kg_write({
                "entities": entity_list,
                "relationships": []
            })
            
            return MCPResponse(
                success=True,
                result={
                    "entities_extracted": len(entities),
                    "memory_updated": write_response.success
                },
                metadata={"skill": "update_memory"}
            )
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
            return MCPResponse(success=False, result=None, error=str(e))
    
    def _handle_write_content(self, parameters: Dict[str, Any]) -> MCPResponse:
        """Handle content writing skill"""
        try:
            topic = parameters.get("topic")
            include_evidence = parameters.get("include_evidence", True)
            
            # Search knowledge graph
            kg_response = self._handle_kg_search({"query": topic})
            
            # Search web for evidence
            evidence = []
            if include_evidence:
                web_response = self._handle_web_search({
                    "query": topic,
                    "max_results": 3
                })
                evidence = web_response.result if web_response.success else []
            
            # Generate content structure
            content = {
                "topic": topic,
                "knowledge_graph_data": kg_response.result,
                "evidence": evidence,
                "summary": f"Medical information about {topic}"
            }
            
            return MCPResponse(
                success=True,
                result=content,
                metadata={"skill": "write_content"}
            )
        except Exception as e:
            logger.error(f"Content writing failed: {e}")
            return MCPResponse(success=False, result=None, error=str(e))
    
    # Public API
    
    def process_request(self, request: MCPRequest) -> MCPResponse:
        """
        Process MCP request.
        Routes to appropriate tool or skill handler.
        """
        # Check if it's a tool
        if request.tool in self.tools:
            tool = self.tools[request.tool]
            logger.info(f"Processing tool request: {tool.name}")
            return tool.handler(request.parameters)
        
        # Check if it's a skill
        if request.tool in self.skills:
            skill = self.skills[request.tool]
            logger.info(f"Processing skill request: {skill.name}")
            return skill.handler(request.parameters)
        
        # Unknown tool/skill
        return MCPResponse(
            success=False,
            result=None,
            error=f"Unknown tool or skill: {request.tool}"
        )
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        return [
            {
                "name": tool.name,
                "type": tool.tool_type.value,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """List available skills"""
        return [
            {
                "name": skill.name,
                "type": skill.skill_type.value,
                "description": skill.description,
                "tools_required": skill.tools_required
            }
            for skill in self.skills.values()
        ]
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities"""
        return {
            "tools": self.list_tools(),
            "skills": self.list_skills(),
            "version": "1.0.0",
            "protocol": "MCP"
        }
