"""
Bootstrap Game Dialog Generator

A Python package for autonomously completing dialogue trees using local LLMs.
"""

__version__ = "1.0.0"
__author__ = "Bootstrap Game Dialog Generator Team"
__description__ = "Autonomous Dialogue Tree Completion Script Using a Local LLM"

from .dialogue_tree import DialogueNode, DialogueTree, DialogueTreeManager
from .llm_integration import NodeGenerator, OllamaClient, PromptGenerator

__all__ = [
    "DialogueTree",
    "DialogueTreeManager",
    "DialogueNode",
    "OllamaClient",
    "NodeGenerator",
    "PromptGenerator",
]
