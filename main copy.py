#!/usr/bin/env python3
"""
Business Agentic Workflow System
A foundational multi-agent orchestration platform for business processes
Uses Qwen3:8B via Ollama for local AI processing
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    id: str
    type: str
    payload: Dict[str, Any]
    priority: Priority = Priority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    assigned_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class Agent(ABC):
    """Base class for all agents in the workflow system"""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.is_busy = False
        self.task_history: List[Task] = []
        
    @abstractmethod
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a task and return results"""
        pass
    
    def can_handle_task(self, task_type: str) -> bool:
        """Check if agent can handle a specific task type"""
        return task_type in self.capabilities

class OllamaClient:
    """Client for interacting with Ollama local models - Windows compatible"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "qwen2.5:7b"  # Default to Qwen2.5:7b (more stable than 3:8b)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _sync_generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Synchronous generate method for threading"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result["response"]
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out")
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}")
        
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text using Ollama - async wrapper around sync call"""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self.executor, 
                self._sync_generate, 
                prompt, 
                system_prompt,
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            raise

# Specialized Business Agents

class DocumentAnalysisAgent(Agent):
    """Agent specialized in document analysis and extraction"""
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__(
            name="DocumentAnalysisAgent",
            capabilities=["document_analysis", "text_extraction", "content_summary"]
        )
        self.ollama = ollama_client
        
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process document analysis tasks"""
        content = task.payload.get("content", "")
        analysis_type = task.payload.get("analysis_type", "summary")
        
        if analysis_type == "summary":
            prompt = f"""
            Please provide a concise business summary of the following document:
            
            {content}
            
            Focus on:
            1. Key points and main topics
            2. Important dates, numbers, and entities
            3. Action items or decisions
            4. Potential risks or opportunities
            
            Format as structured JSON with these fields: summary, key_points, entities, action_items
            """
            
        elif analysis_type == "extract_entities":
            prompt = f"""
            Extract key business entities from this document:
            
            {content}
            
            Extract: names, companies, dates, amounts, locations, products/services
            Format as JSON with entity_type: [list of entities]
            """
            
        elif analysis_type == "sentiment":
            prompt = f"""
            Analyze the sentiment and tone of this business document:
            
            {content}
            
            Provide: overall_sentiment (positive/neutral/negative), confidence_score (0-1), 
            key_phrases, and business_implications
            Format as JSON.
            """
        
        try:
            response = await self.ollama.generate(
                prompt,
                system_prompt="You are a business document analysis expert. Always respond with valid JSON."
            )
            
            # Try to parse as JSON, fallback to text if needed
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                result = {"analysis": response, "format": "text"}
                
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "result": result,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }

class CustomerServiceAgent(Agent):
    """Agent for customer service inquiries and routing"""
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__(
            name="CustomerServiceAgent", 
            capabilities=["customer_inquiry", "ticket_routing", "response_generation"]
        )
        self.ollama = ollama_client
        
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process customer service tasks"""
        inquiry = task.payload.get("inquiry", "")
        customer_info = task.payload.get("customer_info", {})
        
        # Classify and route the inquiry
        classification_prompt = f"""
        Classify this customer inquiry and suggest routing:
        
        Customer: {customer_info.get('name', 'Unknown')}
        Inquiry: {inquiry}
        
        Classify into one of: billing, technical_support, sales, general_inquiry, complaint
        Determine urgency: low, medium, high, critical
        Suggest department routing and initial response
        
        Format as JSON with: category, urgency, suggested_department, confidence_score, initial_response
        """
        
        try:
            response = await self.ollama.generate(
                classification_prompt,
                system_prompt="You are a customer service routing expert. Always respond with valid JSON."
            )
            
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                result = {"classification": response, "format": "text"}
            
            return {
                "status": "success",
                "customer_id": customer_info.get("id"),
                "classification": result,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }

class DataProcessingAgent(Agent):
    """Agent for data analysis and processing tasks"""
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__(
            name="DataProcessingAgent",
            capabilities=["data_analysis", "pattern_detection", "report_generation"]
        )
        self.ollama = ollama_client
        
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process data analysis tasks"""
        data = task.payload.get("data", [])
        analysis_type = task.payload.get("analysis_type", "summary")
        
        # Convert data to string format for LLM processing
        data_str = json.dumps(data, indent=2)
        
        if analysis_type == "trends":
            prompt = f"""
            Analyze this business data for trends and patterns:
            
            {data_str}
            
            Identify:
            1. Key trends over time
            2. Anomalies or outliers
            3. Business insights
            4. Recommendations
            
            Format as JSON with: trends, anomalies, insights, recommendations
            """
            
        elif analysis_type == "summary":
            prompt = f"""
            Provide a business summary of this data:
            
            {data_str}
            
            Include: key_metrics, notable_patterns, business_implications
            Format as JSON.
            """
            
        try:
            response = await self.ollama.generate(
                prompt,
                system_prompt="You are a business data analyst. Always respond with valid JSON."
            )
            
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                result = {"analysis": response, "format": "text"}
                
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "data_points": len(data) if isinstance(data, list) else 1,
                "result": result,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }

class DecisionSupportAgent(Agent):
    """Agent for business decision support and recommendations"""
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__(
            name="DecisionSupportAgent",
            capabilities=["decision_analysis", "risk_assessment", "recommendation_generation"]
        )
        self.ollama = ollama_client
        
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process decision support tasks"""
        scenario = task.payload.get("scenario", "")
        options = task.payload.get("options", [])
        criteria = task.payload.get("criteria", [])
        
        prompt = f"""
        Analyze this business decision scenario:
        
        Scenario: {scenario}
        
        Options to consider:
        {json.dumps(options, indent=2)}
        
        Decision criteria:
        {json.dumps(criteria, indent=2)}
        
        Provide:
        1. Risk assessment for each option
        2. Cost-benefit analysis
        3. Recommended option with reasoning
        4. Implementation considerations
        5. Potential alternatives
        
        Format as JSON with: risk_assessment, cost_benefit, recommendation, implementation, alternatives
        """
        
        try:
            response = await self.ollama.generate(
                prompt,
                system_prompt="You are a strategic business advisor. Provide balanced, analytical recommendations. Always respond with valid JSON."
            )
            
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                result = {"analysis": response, "format": "text"}
                
            return {
                "status": "success",
                "scenario": scenario,
                "options_analyzed": len(options),
                "result": result,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }

class WorkflowOrchestrator:
    """Main orchestrator for the agentic workflow system"""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_client = OllamaClient(ollama_base_url)
        self.agents: List[Agent] = []
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.is_running = False
        
        # Initialize default agents
        self._initialize_agents()
        
        # Test Ollama connection on startup
        self._test_ollama_connection()
        
    def _initialize_agents(self):
        """Initialize all business agents"""
        self.agents = [
            DocumentAnalysisAgent(self.ollama_client),
            CustomerServiceAgent(self.ollama_client),
            DataProcessingAgent(self.ollama_client),
            DecisionSupportAgent(self.ollama_client)
        ]
        logger.info(f"Initialized {len(self.agents)} agents")
        
    def _test_ollama_connection(self):
        """Test connection to Ollama"""
        try:
            response = requests.get(f"{self.ollama_client.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                logger.info(f"Connected to Ollama. Available models: {available_models}")
                
                # Check if our preferred model is available
                if self.ollama_client.model not in available_models:
                    logger.warning(f"Preferred model {self.ollama_client.model} not found. Available: {available_models}")
                    if available_models:
                        self.ollama_client.model = available_models[0]
                        logger.info(f"Using model: {self.ollama_client.model}")
                    else:
                        logger.error("No models available in Ollama!")
            else:
                logger.error(f"Ollama connection test failed: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Make sure it's running on localhost:11434")
            logger.info("To start Ollama, run: ollama serve")
        except Exception as e:
            logger.error(f"Ollama connection test error: {str(e)}")
        
    def add_task(self, task: Task) -> str:
        """Add a task to the queue"""
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
        logger.info(f"Added task {task.id} of type {task.type}")
        return task.id
        
    def get_available_agent(self, task_type: str) -> Optional[Agent]:
        """Find an available agent that can handle the task type"""
        for agent in self.agents:
            if not agent.is_busy and agent.can_handle_task(task_type):
                return agent
        return None
        
    async def process_single_task(self, task: Task) -> None:
        """Process a single task"""
        agent = self.get_available_agent(task.type)
        if not agent:
            logger.warning(f"No available agent for task {task.id} of type {task.type}")
            return
            
        logger.info(f"Assigning task {task.id} to {agent.name}")
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_agent = agent.name
        agent.is_busy = True
        
        try:
            result = await agent.process_task(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Move to completed tasks
            if task in self.task_queue:
                self.task_queue.remove(task)
            self.completed_tasks.append(task)
            agent.task_history.append(task)
            
            logger.info(f"Task {task.id} completed by {agent.name}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            logger.error(f"Task {task.id} failed: {str(e)}")
            
        finally:
            agent.is_busy = False
            
    async def run_workflow(self, max_concurrent: int = 3) -> None:
        """Run the workflow system"""
        self.is_running = True
        logger.info("Starting workflow orchestrator")
        
        while self.is_running:
            if not self.task_queue:
                await asyncio.sleep(1)
                continue
                
            # Get pending tasks that can be processed
            pending_tasks = [t for t in self.task_queue if t.status == TaskStatus.PENDING]
            
            # Limit concurrent processing
            available_agents = [a for a in self.agents if not a.is_busy]
            concurrent_limit = min(max_concurrent, len(available_agents))
            
            tasks_to_process = pending_tasks[:concurrent_limit]
            
            if tasks_to_process:
                await asyncio.gather(*[
                    self.process_single_task(task) for task in tasks_to_process
                ])
            else:
                await asyncio.sleep(0.5)
                
    def stop_workflow(self):
        """Stop the workflow system"""
        self.is_running = False
        logger.info("Stopping workflow orchestrator")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        agent_status = {}
        for agent in self.agents:
            agent_status[agent.name] = {
                "busy": agent.is_busy,
                "capabilities": agent.capabilities,
                "tasks_completed": len(agent.task_history)
            }
            
        return {
            "is_running": self.is_running,
            "tasks_pending": len([t for t in self.task_queue if t.status == TaskStatus.PENDING]),
            "tasks_in_progress": len([t for t in self.task_queue if t.status == TaskStatus.IN_PROGRESS]),
            "tasks_completed": len(self.completed_tasks),
            "agents": agent_status
        }

# Utility functions for common business workflows

def create_document_analysis_task(content: str, analysis_type: str = "summary") -> Task:
    """Create a document analysis task"""
    return Task(
        id=f"doc_analysis_{int(time.time())}",
        type="document_analysis",
        payload={
            "content": content,
            "analysis_type": analysis_type
        }
    )

def create_customer_service_task(inquiry: str, customer_info: Dict[str, Any]) -> Task:
    """Create a customer service task"""
    return Task(
        id=f"customer_service_{int(time.time())}",
        type="customer_inquiry", 
        payload={
            "inquiry": inquiry,
            "customer_info": customer_info
        }
    )

def create_data_analysis_task(data: List[Dict], analysis_type: str = "summary") -> Task:
    """Create a data analysis task"""
    return Task(
        id=f"data_analysis_{int(time.time())}",
        type="data_analysis",
        payload={
            "data": data,
            "analysis_type": analysis_type
        }
    )

def create_decision_support_task(scenario: str, options: List[str], criteria: List[str]) -> Task:
    """Create a decision support task"""
    return Task(
        id=f"decision_support_{int(time.time())}",
        type="decision_analysis",
        payload={
            "scenario": scenario,
            "options": options,
            "criteria": criteria
        }
    )

# Example usage and testing
async def main():
    """Example usage of the business agentic workflow system"""
    
    print("=== Business Agentic Workflow System ===")
    print("Testing Ollama connection...")
    
    # Initialize the orchestrator
    orchestrator = WorkflowOrchestrator()
    
    # Example tasks for different business scenarios
    
    # 1. Document Analysis
    contract_content = """
    Service Agreement between Company A and Company B
    Term: 12 months starting January 1, 2024
    Monthly fee: $50,000
    Payment terms: Net 30
    Renewal: Automatic unless 60-day notice given
    Termination clause: Either party with 30-day written notice
    """
    
    doc_task = create_document_analysis_task(contract_content, "summary")
    orchestrator.add_task(doc_task)
    
    # 2. Customer Service
    customer_inquiry = "I've been charged twice for my monthly subscription and need a refund"
    customer_info = {"id": "CUST001", "name": "John Smith", "tier": "premium"}
    
    service_task = create_customer_service_task(customer_inquiry, customer_info)
    orchestrator.add_task(service_task)
    
    # 3. Data Analysis
    sales_data = [
        {"month": "Jan", "revenue": 100000, "customers": 250},
        {"month": "Feb", "revenue": 120000, "customers": 280},
        {"month": "Mar", "revenue": 95000, "customers": 240},
        {"month": "Apr", "revenue": 150000, "customers": 320}
    ]
    
    data_task = create_data_analysis_task(sales_data, "trends")
    orchestrator.add_task(data_task)
    
    # 4. Decision Support
    decision_scenario = "Should we expand to the European market?"
    options = ["Expand immediately", "Pilot program first", "Wait 12 months", "Partner with local company"]
    criteria = ["Cost", "Risk", "Time to market", "Resource requirements", "Potential ROI"]
    
    decision_task = create_decision_support_task(decision_scenario, options, criteria)
    orchestrator.add_task(decision_task)
    
    print("\nStarting workflow processing...")
    print(f"Initial status: {json.dumps(orchestrator.get_status(), indent=2)}")
    
    # Run for a limited time for demo
    workflow_task = asyncio.create_task(orchestrator.run_workflow())
    
    # Wait for tasks to complete (with timeout)
    completed_count = 0
    for i in range(60):  # Wait up to 60 seconds
        await asyncio.sleep(1)
        status = orchestrator.get_status()
        
        current_completed = status["tasks_completed"]
        if current_completed > completed_count:
            completed_count = current_completed
            print(f"Tasks completed: {completed_count}/4")
        
        if status["tasks_pending"] == 0 and status["tasks_in_progress"] == 0:
            print("All tasks completed!")
            break
            
        if i % 10 == 0 and i > 0:  # Print status every 10 seconds
            print(f"Status update: Pending={status['tasks_pending']}, In Progress={status['tasks_in_progress']}, Completed={status['tasks_completed']}")
    
    orchestrator.stop_workflow()
    workflow_task.cancel()
    
    # Print results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    for i, task in enumerate(orchestrator.completed_tasks, 1):
        print(f"\n[{i}] Task: {task.type.upper()}")
        print(f"    ID: {task.id}")
        print(f"    Status: {task.status.value}")
        print(f"    Agent: {task.assigned_agent}")
        print(f"    Duration: {(task.completed_at - task.created_at).total_seconds():.1f}s")
        
        if task.result and task.result.get("status") == "success":
            result = task.result.get("result", {})
            if isinstance(result, dict):
                print(f"    Result Summary:")
                for key, value in result.items():
                    if isinstance(value, (str, int, float)):
                        print(f"      {key}: {value}")
                    elif isinstance(value, list) and len(value) <= 3:
                        print(f"      {key}: {value}")
                    else:
                        print(f"      {key}: [complex data]")
            else:
                print(f"    Result: {str(result)[:200]}...")
        elif task.error:
            print(f"    Error: {task.error}")
        else:
            print(f"    Result: {task.result}")

if __name__ == "__main__":
    # Set up Windows-compatible event loop
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())