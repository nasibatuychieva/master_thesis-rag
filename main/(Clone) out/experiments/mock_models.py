from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class MockMessage:
    """Mock message model that mimics MessageModel without SQLAlchemy."""
    prompt: str
    answer: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    id: Optional[int] = None


@dataclass
class MockConversation:
    """Mock conversation model that mimics ConversationModel without
    SQLAlchemy."""
    messages: List[MockMessage] = field(default_factory=list)
    id: Optional[int] = None

    def append(self, message: MockMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
