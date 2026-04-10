
"""State Definitions and Pydantic Schemas for Research Scoping.

This defines the state objects and structured schemas used for
the research agent scoping workflow, including researcher state management and output schemas.
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence, Dict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== STATE DEFINITIONS =====

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    start_time: float = 0.0
    target_language: Optional[str] = None

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """

    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Draft research report
    draft_report: str
    # Final formatted research report
    final_report: str
    # Start time of the research process
    start_time: float = 0.0
    # Canonical target language for all generated outputs (e.g., "English", "Chinese")
    target_language: Optional[str] = None
    # Set to True when circuit breaker aborts research (e.g., API quota exhausted)
    aborted: bool = False
    # Secondary subtopic reports: list of dicts {"title": str, "content": str}
    secondary_reports: Annotated[list[dict], operator.add] = []
    # Pending subtopic briefs from evaluation (used to pass to generation node)
    pending_subtopic_briefs: list[dict] = []

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )
    target_language: str = Field(
        description="The language that the report should be written in, based on the language of the user's messages. Unless instructed otherwise, you should write the report in the same langauge as the user's question.",
    )

class DraftReport(BaseModel):
    """Schema for structured draft report generation."""

    draft_report: str = Field(
        description="A draft report that will be used to guide the research.",
    )

# ===== SUBTOPIC EVALUATION TOOLS =====

@tool
class GenerateSubtopicReport(BaseModel):
    """Tool for requesting a subtopic report to be generated.

    Call this for each distinct topic that warrants a detailed supplementary report.
    You can call this multiple times for different topics.
    """
    title: str = Field(
        description="Title for the Subtopic Report (e.g., 'Detailed Analysis of Stock XYZ').",
    )
    generation_brief: str = Field(
        description="Instructions for generating this report. Describe what information to extract from the research notes.",
    )

@tool
class EndSubtopicEvaluation(BaseModel):
    """Tool for indicating that subtopic evaluation is complete.

    Call this when you have finished evaluating and either:
    - Have already called GenerateSubtopicReport for all needed topics, OR
    - Determined that no subtopic reports are necessary
    """
    pass

