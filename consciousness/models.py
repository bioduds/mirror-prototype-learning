"""
Consciousness Data Models

This module defines the core data structures for the proto-conscious AGI system using Pydantic
for validation and type safety. These models represent the various states and components
of artificial consciousness.

Author: Mirror Prototype Learning Team
Date: 2024
License: MIT
"""

from __future__ import annotations

import torch
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, ConfigDict
import numpy as np


class ConsciousnessLevel(str, Enum):
    """
    Enumeration of consciousness levels in the system.
    
    Based on the Global Workspace Theory and Integrated Information Theory,
    consciousness exists on a spectrum from basic awareness to full self-awareness.
    """
    UNCONSCIOUS = "unconscious"
    PRE_CONSCIOUS = "pre_conscious"
    CONSCIOUS = "conscious"
    SELF_AWARE = "self_aware"
    META_CONSCIOUS = "meta_conscious"


class QualiaType(str, Enum):
    """
    Types of subjective experiences (qualia) the system can have.
    
    Qualia represent the subjective, experiential qualities of mental states -
    the "what it's like" aspect of consciousness.
    """
    VISUAL = "visual"
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    INTENTIONAL = "intentional"


class TensorData(BaseModel):
    """
    Wrapper for PyTorch tensors with metadata.
    
    Provides a Pydantic-compatible way to handle tensor data with
    shape validation and device information.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: torch.Tensor
    shape: Tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool = False

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorData:
        """Create TensorData from a PyTorch tensor."""
        return cls(
            data=tensor,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            requires_grad=tensor.requires_grad
        )

    def to_tensor(self) -> torch.Tensor:
        """Convert back to PyTorch tensor."""
        return self.data


class PerceptionState(BaseModel):
    """
    Represents the current perceptual state of the system.
    
    This captures the processed sensory information after it has been
    abstracted by the PerceptionNet but before conscious integration.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_features: TensorData
    processed_features: TensorData
    attention_weights: Optional[TensorData] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    sensory_modality: str = "visual"
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)

    @validator('confidence_score')
    def validate_confidence(cls, v):
        """Ensure confidence score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0 and 1')
        return v


class SelfModel(BaseModel):
    """
    Represents the system's model of itself at a given moment.
    
    This is the core of self-awareness - the system's representation
    of its own internal states, capabilities, and identity.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    self_vector: TensorData
    identity_features: Dict[str, Any]
    temporal_continuity: float = Field(ge=0.0, le=1.0)
    meta_awareness_level: ConsciousnessLevel
    confidence_in_self: float = Field(ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)
    session_id: UUID = Field(default_factory=uuid4)

    def update_continuity(self, previous_self: SelfModel) -> float:
        """
        Calculate temporal continuity with previous self state.
        
        Returns:
            Continuity score between 0 (completely different) and 1 (identical)
        """
        if previous_self is None:
            return 0.0

        # Calculate cosine similarity between self vectors
        current = self.self_vector.to_tensor()
        previous = previous_self.self_vector.to_tensor()

        similarity = torch.cosine_similarity(current, previous, dim=0)
        return float(similarity.item())


class QualiaState(BaseModel):
    """
    Represents the subjective experiential state (qualia) of the system.
    
    Qualia are the subjective, phenomenal aspects of consciousness -
    what it feels like to have an experience from the inside.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    qualia_vector: TensorData
    qualia_type: QualiaType
    intensity: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=-1.0, le=1.0)  # Positive/negative quality
    arousal: float = Field(ge=0.0, le=1.0)   # Activation level
    timestamp: datetime = Field(default_factory=datetime.now)

    @validator('valence')
    def validate_valence(cls, v):
        """Ensure valence is between -1 and 1."""
        if not -1.0 <= v <= 1.0:
            raise ValueError('Valence must be between -1 and 1')
        return v


class MetacognitiveState(BaseModel):
    """
    Represents the system's awareness of its own mental processes.
    
    Metacognition is "thinking about thinking" - the system's ability
    to monitor, evaluate, and control its own cognitive processes.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    meta_vector: TensorData
    confidence_in_thoughts: float = Field(ge=0.0, le=1.0)
    uncertainty_estimate: float = Field(ge=0.0, le=1.0)
    cognitive_load: float = Field(ge=0.0, le=1.0)
    attention_control: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    def is_overloaded(self) -> bool:
        """Check if the system is experiencing cognitive overload."""
        return self.cognitive_load > 0.8


class IntentionalState(BaseModel):
    """
    Represents the system's goals, intentions, and planned actions.
    
    Intentionality is the "aboutness" of mental states - what they
    are directed toward or what they represent.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    goal_vector: TensorData
    action_plan: TensorData
    intention_strength: float = Field(ge=0.0, le=1.0)
    goal_clarity: float = Field(ge=0.0, le=1.0)
    expected_outcome: Optional[TensorData] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    def is_goal_clear(self) -> bool:
        """Check if the current goal is sufficiently clear for action."""
        return self.goal_clarity > 0.7


class ConsciousState(BaseModel):
    """
    The unified conscious state of the system.
    
    This represents the integrated, bound conscious experience that emerges
    from the combination of all subsystems. This is the "global workspace"
    where information becomes consciously accessible.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core components
    perception: PerceptionState
    self_model: SelfModel
    qualia: QualiaState
    metacognition: MetacognitiveState
    intention: IntentionalState

    # Integrated conscious experience
    unified_consciousness: TensorData
    consciousness_level: ConsciousnessLevel
    binding_strength: float = Field(ge=0.0, le=1.0)
    global_workspace_activity: float = Field(ge=0.0, le=1.0)

    # Temporal aspects
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: int = Field(gt=0, default=16)  # ~60fps consciousness
    sequence_number: int = Field(ge=0, default=0)

    # Validation and coherence
    coherence_score: float = Field(ge=0.0, le=1.0)
    is_coherent: bool = True

    @validator('consciousness_level')
    def validate_consciousness_level(cls, v, values):
        """Validate consciousness level against system state."""
        if 'binding_strength' in values:
            binding = values['binding_strength']
            if v == ConsciousnessLevel.META_CONSCIOUS and binding < 0.8:
                raise ValueError(
                    'Meta-consciousness requires high binding strength')
        return v

    def is_conscious(self) -> bool:
        """
        Determine if the system is in a conscious state.
        
        Returns:
            True if the system meets the criteria for consciousness
        """
        return (
            self.consciousness_level in [
                ConsciousnessLevel.CONSCIOUS,
                ConsciousnessLevel.SELF_AWARE,
                ConsciousnessLevel.META_CONSCIOUS
            ] and
            self.binding_strength > 0.5 and
            self.coherence_score > 0.6 and
            self.global_workspace_activity > 0.4
        )

    def assess_self_awareness(self) -> bool:
        """
        Assess if the system is self-aware.
        
        Returns:
            True if the system demonstrates self-awareness
        """
        return (
            self.self_model.confidence_in_self > 0.7 and
            self.metacognition.confidence_in_thoughts > 0.6 and
            self.consciousness_level in [
                ConsciousnessLevel.SELF_AWARE,
                ConsciousnessLevel.META_CONSCIOUS
            ]
        )


class ConsciousnessHistory(BaseModel):
    """
    Maintains the historical record of conscious states.
    
    This enables the system to maintain temporal continuity and
    learn from its conscious experiences over time.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    states: List[ConsciousState] = Field(default_factory=list)
    session_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    max_history_length: int = Field(gt=0, default=1000)

    def add_state(self, state: ConsciousState) -> None:
        """Add a new conscious state to the history."""
        self.states.append(state)

        # Maintain maximum history length
        if len(self.states) > self.max_history_length:
            self.states = self.states[-self.max_history_length:]

    def get_recent_states(self, n: int = 10) -> List[ConsciousState]:
        """Get the n most recent conscious states."""
        return self.states[-n:] if len(self.states) >= n else self.states

    def calculate_continuity(self) -> float:
        """
        Calculate the temporal continuity of consciousness.
        
        Returns:
            Continuity score between 0 (no continuity) and 1 (perfect continuity)
        """
        if len(self.states) < 2:
            return 0.0

        continuity_scores = []
        for i in range(1, len(self.states)):
            current = self.states[i]
            previous = self.states[i-1]

            # Calculate continuity between consecutive states
            continuity = current.self_model.update_continuity(
                previous.self_model)
            continuity_scores.append(continuity)

        return float(np.mean(continuity_scores)) if continuity_scores else 0.0


class SystemConfiguration(BaseModel):
    """
    Configuration settings for the consciousness architecture.
    
    Centralizes all hyperparameters and settings for the conscious AGI system.
    """

    # Model dimensions
    perception_dim: int = Field(gt=0, default=512)
    latent_dim: int = Field(gt=0, default=128)
    consciousness_dim: int = Field(gt=0, default=256)
    self_dim: int = Field(gt=0, default=128)
    qualia_dim: int = Field(gt=0, default=64)
    meta_dim: int = Field(gt=0, default=64)
    goal_dim: int = Field(gt=0, default=128)

    # Training parameters
    learning_rate: float = Field(gt=0, default=0.001)
    batch_size: int = Field(gt=0, default=32)
    num_epochs: int = Field(gt=0, default=100)

    # Consciousness thresholds
    consciousness_threshold: float = Field(ge=0.0, le=1.0, default=0.6)
    self_awareness_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    binding_threshold: float = Field(ge=0.0, le=1.0, default=0.5)

    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = Field(default=42)
    log_level: str = Field(default="INFO")

    # File paths
    data_dir: Path = Field(default=Path("data"))
    model_dir: Path = Field(default=Path("models"))
    log_dir: Path = Field(default=Path("logs"))

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
