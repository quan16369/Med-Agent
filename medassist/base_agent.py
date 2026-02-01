"""
Base Agent class for MedAssist system
All specialized agents inherit from this base class
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    sender: str
    recipient: str
    content: Dict[str, Any]
    timestamp: str
    message_type: str  # 'request', 'response', 'update'


class BaseAgent:
    """Base class for all medical agents"""
    
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        model,
        tokenizer,
        tools: Optional[List] = None
    ):
        """
        Initialize base agent
        
        Args:
            name: Agent name
            role: Agent role/type
            system_prompt: System prompt defining agent behavior
            model: Language model instance
            tokenizer: Tokenizer instance
            tools: List of tools the agent can use
        """
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools or []
        self.memory = []  # Conversation history
        self.state = {}  # Agent state
        
        logger.info(f"Initialized agent: {self.name} ({self.role})")
    
    def add_to_memory(self, role: str, content: str):
        """Add message to agent memory"""
        self.memory.append({
            "role": role,
            "content": content
        })
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response using the language model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        try:
            # Build full prompt with system prompt
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            
            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def use_tool(self, tool_name: str, tool_input: Dict) -> Any:
        """
        Use a tool from the agent's toolbox
        
        Args:
            tool_name: Name of the tool to use
            tool_input: Input parameters for the tool
            
        Returns:
            Tool execution result
        """
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = tool.run(tool_input)
                    logger.info(f"Tool {tool_name} executed successfully")
                    return result
                except Exception as e:
                    logger.error(f"Error using tool {tool_name}: {e}")
                    return {"error": str(e)}
        
        logger.warning(f"Tool {tool_name} not found")
        return {"error": f"Tool {tool_name} not available"}
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input and generate agent response
        This should be overridden by specialized agents
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Response dictionary with agent findings
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def reset(self):
        """Reset agent state and memory"""
        self.memory = []
        self.state = {}
        logger.info(f"Reset agent: {self.name}")
    
    def get_state(self) -> Dict:
        """Get current agent state"""
        return {
            "name": self.name,
            "role": self.role,
            "memory_length": len(self.memory),
            "state": self.state
        }
    
    def parse_action(self, response: str) -> Optional[Dict]:
        """
        Parse agent response to extract action and tool calls
        Uses ReAct pattern: Thought -> Action -> Observation
        
        Args:
            response: Agent's text response
            
        Returns:
            Dictionary with action information or None
        """
        try:
            # Look for action patterns
            if "Action:" in response and "Action Input:" in response:
                lines = response.split('\n')
                action = None
                action_input = None
                
                for i, line in enumerate(lines):
                    if line.strip().startswith("Action:"):
                        action = line.split("Action:")[1].strip()
                    elif line.strip().startswith("Action Input:"):
                        # Get everything after Action Input:
                        action_input = '\n'.join(lines[i:]).split("Action Input:")[1].strip()
                        break
                
                if action and action_input:
                    try:
                        # Try to parse as JSON
                        parsed_input = json.loads(action_input)
                    except json.JSONDecodeError:
                        # Use as string if not JSON
                        parsed_input = action_input
                    
                    return {
                        "action": action,
                        "action_input": parsed_input
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing action: {e}")
            return None


class AgentTools:
    """Collection of tools that agents can use"""
    
    @staticmethod
    def calculator(expression: str) -> str:
        """
        Simple calculator tool for medical calculations
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result as string
        """
        try:
            # Safety: only allow specific functions
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def bmi_calculator(weight_kg: float, height_m: float) -> Dict:
        """
        Calculate Body Mass Index
        
        Args:
            weight_kg: Weight in kilograms
            height_m: Height in meters
            
        Returns:
            BMI value and category
        """
        bmi = weight_kg / (height_m ** 2)
        
        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal weight"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
        
        return {
            "bmi": round(bmi, 1),
            "category": category
        }
    
    @staticmethod
    def egfr_calculator(
        creatinine_mg_dl: float,
        age: int,
        is_female: bool,
        is_black: bool = False
    ) -> Dict:
        """
        Calculate estimated Glomerular Filtration Rate (eGFR) using CKD-EPI equation
        
        Args:
            creatinine_mg_dl: Serum creatinine in mg/dL
            age: Age in years
            is_female: Whether patient is female
            is_black: Whether patient is Black/African American
            
        Returns:
            eGFR value and CKD stage
        """
        # CKD-EPI equation
        kappa = 0.7 if is_female else 0.9
        alpha = -0.329 if is_female else -0.411
        
        min_val = min(creatinine_mg_dl / kappa, 1)
        max_val = max(creatinine_mg_dl / kappa, 1)
        
        egfr = 141 * (min_val ** alpha) * (max_val ** -1.209) * (0.993 ** age)
        
        if is_female:
            egfr *= 1.018
        if is_black:
            egfr *= 1.159
        
        # Determine CKD stage
        if egfr >= 90:
            stage = "G1 (Normal or high)"
        elif egfr >= 60:
            stage = "G2 (Mildly decreased)"
        elif egfr >= 45:
            stage = "G3a (Mild to moderately decreased)"
        elif egfr >= 30:
            stage = "G3b (Moderately to severely decreased)"
        elif egfr >= 15:
            stage = "G4 (Severely decreased)"
        else:
            stage = "G5 (Kidney failure)"
        
        return {
            "egfr": round(egfr, 1),
            "stage": stage,
            "unit": "mL/min/1.73m²"
        }
