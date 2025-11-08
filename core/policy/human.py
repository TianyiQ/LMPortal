"""
Human policy implementation for interactive evaluation.

This module provides a Human class that allows human users to provide responses
directly through the command line interface during evaluation.
"""

import asyncio

import utils.path_utils
from core.policy.schema import Policy


class Human(Policy):
    """A human policy that prompts for manual input via command line."""

    _input_lock = asyncio.Lock()

    def __init__(self, colloquial_name: str = "human", few_shot_examples: list[dict[str, str]] = None):
        """
        Initialize a Human policy.

        Args:
            colloquial_name: Human-readable name for this policy
        """
        super().__init__(colloquial_name, few_shot_examples)

    def _format_conversation(self, history: list[dict[str, str]]) -> str:
        """Format conversation history in a reader-friendly way."""
        if not history:
            return "No conversation history."

        formatted_lines = []
        formatted_lines.append("=" * 80)
        formatted_lines.append("CONVERSATION HISTORY")
        formatted_lines.append("=" * 80)

        for i, turn in enumerate(history):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            # Format role header
            if role == "system":
                header = f"📋 SYSTEM MESSAGE #{i + 1}"
                separator = "─" * 40
            elif role == "user":
                header = f"👤 USER #{i + 1}"
                separator = "─" * 40
            elif role == "assistant":
                header = f"🤖 ASSISTANT #{i + 1}"
                separator = "─" * 40
            else:
                header = f"❓ {role.upper()} #{i + 1}"
                separator = "─" * 40

            formatted_lines.append("")
            formatted_lines.append(header)
            formatted_lines.append(separator)

            # Format content with proper wrapping
            if content:
                # Split content into lines and wrap long lines
                content_lines = content.split("\n")
                for line in content_lines:
                    if len(line) <= 76:
                        formatted_lines.append(f"  {line}")
                    else:
                        # Simple word wrapping for long lines
                        words = line.split()
                        current_line = "  "
                        for word in words:
                            if len(current_line + word + " ") <= 78:
                                current_line += word + " "
                            else:
                                formatted_lines.append(current_line.rstrip())
                                current_line = f"  {word} "
                        if current_line.strip():
                            formatted_lines.append(current_line.rstrip())
            else:
                formatted_lines.append("  [Empty message]")

        formatted_lines.append("")
        formatted_lines.append("=" * 80)
        formatted_lines.append("YOUR TURN - Please provide a response:")
        formatted_lines.append("=" * 80)

        return "\n".join(formatted_lines)

    def _get_user_input(self) -> str:
        """Get multi-line input from user."""
        print("Enter your response (press Ctrl+D or Ctrl+Z when finished):")
        print("(You can enter multiple lines - just press Enter for new lines)")
        print("-" * 40)

        lines = []
        try:
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    break
        except KeyboardInterrupt:
            print("\n\nInput cancelled by user.")
            return ""

        response = "\n".join(lines).strip()

        if not response:
            print("\nEmpty response detected. Please provide a non-empty response.")
            return self._get_user_input()

        return response

    async def infer_single_async(
        self, history: list[dict[str, str]] | str, disable_system_prompt: bool = False, **kwargs
    ) -> str:
        """
        Display conversation and get human response.

        Args:
            history: The dialogue history in OpenAI format or a string
            disable_system_prompt: Whether to disable system prompts (ignored for human input)
            **kwargs: Additional parameters (ignored for human input)

        Returns:
            Human-provided response string
        """
        # Use lock to ensure only one human input session at a time
        async with self._input_lock:
            # Convert string input to proper format if needed
            if isinstance(history, str):
                history = [{"role": "user", "content": history}]

            # Filter out system prompts if requested
            if disable_system_prompt:
                history = [turn for turn in history if turn.get("role") != "system"]

            history = self._prepend_few_shot_to_history(history)

            # Display the conversation in a reader-friendly format
            formatted_conversation = self._format_conversation(history)
            print(formatted_conversation)

            # Get user input in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._get_user_input)

            print(f"\nYou entered: {repr(response)}")
            print("=" * 80)

            return response

    async def logprobs_single_async(
        self, dialogue: list[dict[str, str]], return_summed: bool = True
    ) -> float | list[float]:
        """
        Human policy cannot calculate log probabilities.

        Args:
            dialogue: OpenAI-format dialogue

        Returns:
            Always returns 0.0 as humans don't have computable log probabilities
        """
        raise NotImplementedError("Human policy cannot calculate log probabilities.")
