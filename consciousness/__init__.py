"""
Consciousness Package

A comprehensive neural architecture for artificial consciousness and proto-AGI.

This package implements the theoretical framework for consciousness emergence
through recursive self-abstraction and mirror neural networks.

Modules:
    models: Pydantic data models for consciousness states
    networks: Neural network architectures for consciousness components
    integrator: Main consciousness integration system
    tests: Consciousness validation and testing suite
    utils: Utility functions and helpers

Author: Mirror Prototype Learning Team
Date: 2024
License: MIT
"""

from .models import *
from .networks import *

__version__ = "1.0.0"
__author__ = "Mirror Prototype Learning Team"
__all__ = [
    "ConsciousnessLevel",
    "QualiaType",
    "TensorData",
    "PerceptionState",
    "SelfModel",
    "QualiaState",
    "MetacognitiveState",
    "IntentionalState",
    "ConsciousState",
    "ConsciousnessHistory",
    "SystemConfiguration"
]
