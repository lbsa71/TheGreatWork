#!/usr/bin/env python3
"""
Tests for dialogue tree validation functionality.
"""

from src.dialogue_tree import DialogueTree


class TestDialogueTreeValidation:
    """Tests for dialogue tree validation and fixing."""

    def test_validate_and_fix_tree_complete(self) -> None:
        """Test validation on a complete, valid tree."""
        nodes = {
            "start": {
                "situation": "The king is dead.",
                "choices": [
                    {
                        "text": "Mourn publicly",
                        "next": "node1",
                        "effects": {"loyalty": 10},
                    },
                    {
                        "text": "Seize the throne",
                        "next": "node2",
                        "effects": {"ambition": 15},
                    },
                ],
            },
            "node1": {
                "situation": "You mourn publicly.",
                "choices": [
                    {"text": "Continue mourning", "next": None},
                ],
            },
            "node2": None,
        }
        params = {"loyalty": 50, "ambition": 30}

        tree = DialogueTree(nodes, params)
        tree.validate_and_fix_tree()

        # Tree should remain unchanged since it's already valid
        assert tree.nodes["start"]["situation"] == "The king is dead."
        assert len(tree.nodes["start"]["choices"]) == 2
        assert tree.nodes["node1"]["situation"] == "You mourn publicly."
        assert tree.nodes["node2"] is None
        assert tree.params["loyalty"] == 50
        assert tree.params["ambition"] == 30

    def test_create_missing_referenced_nodes(self) -> None:
        """Test that missing referenced nodes are created as null."""
        nodes = {
            "start": {
                "situation": "The king is dead.",
                "choices": [
                    {"text": "Mourn publicly", "next": "missing_node1"},
                    {"text": "Seize the throne", "next": "missing_node2"},
                ],
            },
        }
        params = {"loyalty": 50}

        tree = DialogueTree(nodes, params)
        tree.validate_and_fix_tree()

        # Missing nodes should be created as null
        assert "missing_node1" in tree.nodes
        assert tree.nodes["missing_node1"] is None
        assert "missing_node2" in tree.nodes
        assert tree.nodes["missing_node2"] is None

    def test_remove_orphaned_nodes(self) -> None:
        """Test that orphaned nodes are removed."""
        nodes = {
            "start": {
                "situation": "The king is dead.",
                "choices": [
                    {"text": "Mourn publicly", "next": "node1"},
                ],
            },
            "node1": {
                "situation": "You mourn publicly.",
                "choices": [
                    {"text": "Continue", "next": None},
                ],
            },
            "orphan1": {
                "situation": "This node is orphaned.",
                "choices": [],
            },
            "orphan2": None,  # Orphaned null node
        }
        params = {"loyalty": 50}

        tree = DialogueTree(nodes, params)
        tree.validate_and_fix_tree()

        # Orphaned nodes should be removed
        assert "start" in tree.nodes  # Root node should remain
        assert "node1" in tree.nodes  # Referenced node should remain
        assert "orphan1" not in tree.nodes  # Orphaned node should be removed
        assert "orphan2" not in tree.nodes  # Orphaned null node should be removed

    def test_extract_and_add_missing_params(self) -> None:
        """Test that missing effect parameters are added to root params."""
        nodes = {
            "start": {
                "situation": "The king is dead.",
                "choices": [
                    {
                        "text": "Mourn publicly",
                        "next": "node1",
                        "effects": {"loyalty": 10, "wisdom": 5},
                    },
                    {
                        "text": "Seize the throne",
                        "next": "node2",
                        "effects": {"ambition": 15, "charisma": 8},
                    },
                ],
            },
            "node1": {
                "situation": "You mourn publicly.",
                "choices": [
                    {"text": "Continue", "next": None, "effects": {"reputation": 3}},
                ],
            },
            "node2": None,
        }
        params = {"loyalty": 50}  # Missing: wisdom, ambition, charisma, reputation

        tree = DialogueTree(nodes, params)
        tree.validate_and_fix_tree()

        # Missing parameters should be added with default value 0
        assert tree.params["loyalty"] == 50  # Existing param unchanged
        assert tree.params["wisdom"] == 0  # New param added
        assert tree.params["ambition"] == 0  # New param added
        assert tree.params["charisma"] == 0  # New param added
        assert tree.params["reputation"] == 0  # New param added

    def test_fix_invalid_node_structure(self) -> None:
        """Test that invalid nodes are converted to null or removed."""
        nodes = {
            "start": {
                "situation": "The king is dead.",
                "choices": [
                    {"text": "Mourn publicly", "next": "valid_node"},
                    {"text": "Invalid choice", "next": "invalid_node1"},
                    {"text": "Another choice", "next": "invalid_node2"},
                ],
            },
            "valid_node": {
                "situation": "Valid node.",
                "choices": [
                    {"text": "Continue", "next": None},
                ],
            },
            "invalid_node1": {
                # Missing situation
                "choices": [
                    {"text": "Invalid", "next": None},
                ],
            },
            "invalid_node2": {
                "situation": "Has situation",
                # Missing choices
            },
            "invalid_node3": "not a dict",  # Not a dictionary - orphaned
            "invalid_node4": 123,  # Not a dictionary - orphaned
        }
        params = {"loyalty": 50}

        tree = DialogueTree(nodes, params)
        tree.validate_and_fix_tree()

        # Valid nodes should remain
        assert tree.nodes["start"]["situation"] == "The king is dead."
        assert tree.nodes["valid_node"]["situation"] == "Valid node."

        # Referenced invalid nodes should be converted to null
        assert tree.nodes["invalid_node1"] is None
        assert tree.nodes["invalid_node2"] is None

        # Unreferenced invalid nodes should be removed entirely
        assert "invalid_node3" not in tree.nodes
        assert "invalid_node4" not in tree.nodes

    def test_remove_extra_fields_from_nodes(self) -> None:
        """Test that extra fields are removed from nodes."""
        nodes = {
            "start": {
                "situation": "The king is dead.",
                "choices": [
                    {"text": "Mourn publicly", "next": "node1"},
                ],
                "extra_field": "should be removed",
                "another_extra": 123,
            },
            "node1": {
                "situation": "You mourn publicly.",
                "choices": [
                    {
                        "text": "Continue",
                        "next": None,
                        "extra_choice_field": "remove me",
                    },
                ],
                "unwanted": "remove this too",
            },
        }
        params = {"loyalty": 50}

        tree = DialogueTree(nodes, params)
        tree.validate_and_fix_tree()

        # Extra fields should be removed
        assert "extra_field" not in tree.nodes["start"]
        assert "another_extra" not in tree.nodes["start"]
        assert "unwanted" not in tree.nodes["node1"]
        assert "extra_choice_field" not in tree.nodes["start"]["choices"][0]

        # Valid fields should remain
        assert tree.nodes["start"]["situation"] == "The king is dead."
        assert tree.nodes["start"]["choices"][0]["text"] == "Mourn publicly"
        assert tree.nodes["start"]["choices"][0]["next"] == "node1"

    def test_fix_invalid_choices(self) -> None:
        """Test that invalid choices are removed."""
        nodes = {
            "start": {
                "situation": "The king is dead.",
                "choices": [
                    {"text": "Valid choice", "next": "node1"},
                    {"next": "node2"},  # Missing text
                    "invalid choice string",  # Not a dict
                    {"text": 123, "next": "node3"},  # text not string
                    {
                        "text": "Another valid",
                        "next": "node4",
                        "effects": {"loyalty": 5},
                    },
                    {
                        "text": "Invalid effects",
                        "next": "node5",
                        "effects": "not a dict",
                    },
                ],
            },
            "node1": None,
            "node4": None,
            "node5": None,
        }
        params = {"loyalty": 50}

        tree = DialogueTree(nodes, params)
        tree.validate_and_fix_tree()

        # Only valid choices should remain
        choices = tree.nodes["start"]["choices"]
        assert len(choices) == 3  # Updated expectation

        # First valid choice
        assert choices[0]["text"] == "Valid choice"
        assert choices[0]["next"] == "node1"

        # Second valid choice with effects
        assert choices[1]["text"] == "Another valid"
        assert choices[1]["next"] == "node4"
        assert choices[1]["effects"] == {"loyalty": 5}

        # Third choice with invalid effects stripped
        assert choices[2]["text"] == "Invalid effects"
        assert choices[2]["next"] == "node5"
        assert "effects" not in choices[2]

    def test_preserve_root_nodes(self) -> None:
        """Test that root nodes are preserved even if not referenced."""
        nodes = {
            "start": {
                "situation": "The beginning.",
                "choices": [
                    {"text": "Go to node1", "next": "node1"},
                ],
            },
            "node1": {
                "situation": "Middle node.",
                "choices": [
                    {"text": "Continue", "next": None},
                ],
            },
            "isolated_root": {
                "situation": "Another root node.",
                "choices": [
                    {"text": "End", "next": None},
                ],
            },
        }
        params = {"loyalty": 50}

        tree = DialogueTree(nodes, params)
        tree.validate_and_fix_tree()

        # All nodes should be preserved (start and isolated_root are both roots)
        assert "start" in tree.nodes
        assert "node1" in tree.nodes
        assert "isolated_root" in tree.nodes

    def test_empty_effects_handling(self) -> None:
        """Test handling of empty effects objects."""
        nodes = {
            "start": {
                "situation": "The king is dead.",
                "choices": [
                    {"text": "Choice with empty effects", "next": None, "effects": {}},
                    {"text": "Choice without effects", "next": None},
                ],
            },
        }
        params = {"loyalty": 50}

        tree = DialogueTree(nodes, params)
        tree.validate_and_fix_tree()

        # Empty effects should be preserved
        choices = tree.nodes["start"]["choices"]
        assert "effects" in choices[0]
        assert choices[0]["effects"] == {}
        assert "effects" not in choices[1]

        # No new params should be added from empty effects
        assert list(tree.params.keys()) == ["loyalty"]

    def test_complex_validation_scenario(self) -> None:
        """Test a complex scenario with multiple validation issues."""
        nodes = {
            "start": {
                "situation": "Complex scenario.",
                "choices": [
                    {
                        "text": "Go to valid",
                        "next": "valid_node",
                        "effects": {"new_param": 10},
                    },
                    {
                        "text": "Go to missing",
                        "next": "missing_node",
                        "effects": {"another_param": 5},
                    },
                    {
                        "text": "Go to invalid",
                        "next": "invalid_node",
                        "effects": {"third_param": 3},
                    },
                ],
                "extra_field": "remove me",
            },
            "valid_node": {
                "situation": "Valid node.",
                "choices": [
                    {"text": "End", "next": None},
                ],
            },
            "invalid_node": {
                "choices": [],  # Missing situation
            },
            "orphan": {
                "situation": "Nobody references me.",
                "choices": [],
            },
        }
        params = {"loyalty": 50}

        tree = DialogueTree(nodes, params)
        original_node_count = len(tree.nodes)
        tree.validate_and_fix_tree()

        # Check all validation fixes were applied

        # 1. Missing referenced node should be created
        assert "missing_node" in tree.nodes
        assert tree.nodes["missing_node"] is None

        # 2. Orphaned node should be removed
        assert "orphan" not in tree.nodes

        # 3. Invalid referenced node should be nullified
        assert tree.nodes["invalid_node"] is None

        # 4. Effect params should be added
        assert "new_param" in tree.params
        assert tree.params["new_param"] == 0
        assert "another_param" in tree.params
        assert tree.params["another_param"] == 0
        assert "third_param" in tree.params
        assert tree.params["third_param"] == 0

        # 5. Extra fields should be removed
        assert "extra_field" not in tree.nodes["start"]

        # Valid structure should be preserved
        assert tree.nodes["start"]["situation"] == "Complex scenario."
        assert tree.nodes["valid_node"]["situation"] == "Valid node."
