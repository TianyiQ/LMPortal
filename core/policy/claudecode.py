"""
Claude Code policy implementation.

This module provides a ClaudeCode class that interfaces with the Claude Code CLI
for executing research tasks and investigations.
"""

import asyncio
import os
import tempfile

from core.policy.schema import Policy


class ClaudeCode(Policy):
    """
    A policy that interfaces with Claude Code CLI for research and investigation tasks.

    Claude Code is a research agent that can execute complex tasks involving
    web search, data analysis, file operations, and other external tools.
    """

    def __init__(
        self, colloquial_name: str = "Claude-Code", timeout: int = 1200, few_shot_examples: list[dict[str, str]] = None
    ):  # 20 minutes default timeout
        """
        Initialize Claude Code policy.

        :param colloquial_name: Human-readable name for this policy
        :param timeout: Timeout in seconds for Claude Code commands (default: 20 minutes)
        :param few_shot_examples: Optional few-shot examples for in-context learning
        """
        super().__init__(colloquial_name, few_shot_examples)
        self.timeout = timeout

    async def infer_single_async(
        self, history: list[dict[str, str]], disable_system_prompt: bool = False, working_dir: str = None, **kwargs
    ) -> str:
        """
        Execute a single research task using Claude Code CLI.

        :param history: Conversation history in OpenAI format
        :param disable_system_prompt: Ignored (Claude Code handles its own system prompts)
        :param working_dir: Optional working directory to use instead of creating a temporary one
        :param kwargs: Additional arguments (timeout can be overridden here)
        :return: Claude Code's response
        """
        history = self._prepend_few_shot_to_history(history)

        # Extract the user's request from the conversation
        user_messages = [msg for msg in history if msg["role"] == "user"]
        if not user_messages:
            raise ValueError("No user messages found in conversation history")

        # Use the last user message as the primary request
        request = user_messages[-1]["content"]

        # Add context from previous messages if available
        if len(history) > 1:
            context_messages = []
            for msg in history[:-1]:  # All messages except the last one
                if msg["role"] == "system":
                    context_messages.append(f"System: {msg['content']}")
                elif msg["role"] == "user":
                    context_messages.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    context_messages.append(f"Assistant: {msg['content']}")

            if context_messages:
                context = "\n".join(context_messages)
                request = f"Previous context:\n{context}\n\nCurrent request:\n{request}"

        # Get timeout from kwargs or use default
        timeout = kwargs.get("timeout", self.timeout)

        # Execute Claude Code
        return await self._execute_claude_code(request, timeout, working_dir)

    async def _execute_claude_code(self, request: str, timeout: int, working_dir: str = None) -> str:
        """
        Execute Claude Code CLI with the given request.

        :param request: The request/prompt to send to Claude Code
        :param timeout: Timeout in seconds
        :param working_dir: Optional working directory to use instead of creating a temporary one
        :return: Claude Code's response
        """
        try:
            # Use provided working directory or create a temporary one
            if working_dir:
                # Use the provided directory, create if it doesn't exist
                os.makedirs(working_dir, exist_ok=True)
                temp_dir = working_dir
                print(f"Claude Code working directory (provided): {temp_dir}")
            else:
                # Create a temporary working directory (don't auto-delete for inspection)
                temp_dir = tempfile.mkdtemp(prefix="claude_code_")
                print(f"Claude Code working directory (temporary): {temp_dir}")

            try:
                temp_dir_path = os.path.abspath(temp_dir)

                # Create result file path
                result_file = os.path.join(temp_dir_path, "result.txt")

                # Enhanced request with safety instructions
                enhanced_request = f"""
You are working in a temporary directory: {temp_dir_path}

IMPORTANT SAFETY INSTRUCTIONS:
- Only work within this temporary directory
- Do NOT modify, create, or delete files outside this directory
- Do NOT run commands that could affect the system outside this directory

TASK:
{request}

OUTPUT INSTRUCTIONS:
- Write your final results/findings to the file: {result_file}
- Include all important information in this file, including as much raw data/raw analysis stats as is feasible, and also non-cherry-picked examples.
- Use clear formatting for easy reading, including lists, YAML-formatted tables, etc.
"""

                # Construct Claude Code command using print mode (-p) to avoid interactivity
                cmd = [
                    "claude",
                    "-p",
                    enhanced_request,
                    "--dangerously-skip-permissions",  # Skip permission prompts for automation
                    "--verbose",  # Show full turn-by-turn output for visibility
                    "--output-format",
                    "text",  # Ensure text output
                    # Note: No max-turns limit to allow full task completion
                ]

                # Execute the command in the temporary directory
                # Don't capture stdout/stderr so output is visible in real-time
                print(f"Executing command: {' '.join(cmd)}")
                print(f"Working directory: {temp_dir_path}")
                print("=" * 60)

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=temp_dir_path,
                    # Note: Not capturing stdout/stderr to allow real-time visibility
                )

                try:
                    # Wait for process to complete (no stdout/stderr to read)
                    await asyncio.wait_for(process.wait(), timeout=timeout)

                    print("=" * 60)
                    print(f"Claude Code process completed with return code: {process.returncode}")

                    # Try to read the result file first
                    result_content = ""
                    try:
                        if os.path.exists(result_file):
                            with open(result_file, encoding="utf-8") as f:
                                result_content = f.read()
                            print(f"📁 Result file found and read ({len(result_content)} characters)")
                        else:
                            print(f"⚠️ Result file not found at: {result_file}")
                    except Exception as e:
                        print(f"WARNING: Could not read result file: {e}")

                    # List all files in the working directory for inspection
                    try:
                        files = os.listdir(temp_dir_path)
                        if files:
                            print("📂 Files created in working directory:")
                            for file in files:
                                file_path = os.path.join(temp_dir_path, file)
                                if os.path.isfile(file_path):
                                    size = os.path.getsize(file_path)
                                    print(f"   {file} ({size} bytes)")
                    except Exception as e:
                        print(f"WARNING: Could not list directory contents: {e}")

                    # Prepare response
                    response_parts = []

                    response_parts.append("=== CLAUDE CODE EXECUTION COMPLETED ===")
                    response_parts.append(f"Working directory preserved at: {temp_dir_path}")
                    response_parts.append(f"Return code: {process.returncode}")

                    if result_content.strip():
                        response_parts.append("=== INVESTIGATION RESULTS ===")
                        response_parts.append(result_content.strip())
                    else:
                        response_parts.append("=== NO RESULTS FILE FOUND ===")
                        response_parts.append("Check the working directory manually for any outputs.")

                    # Check return code
                    if process.returncode != 0:
                        response_parts.append("=== WARNING ===")
                        response_parts.append(f"Claude Code exited with non-zero return code: {process.returncode}")

                    # Return combined response
                    return "\n\n".join(response_parts)

                except TimeoutError as timeout_err:
                    # Kill the process if it times out
                    try:
                        process.kill()
                        await process.wait()
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Claude Code timed out after {timeout} seconds. Working directory preserved at: {temp_dir_path}"
                    ) from timeout_err

            except Exception:
                # Don't clean up temp directory on error so we can inspect it
                print(f"Exception occurred. Working directory preserved at: {temp_dir_path}")
                raise

        except FileNotFoundError as file_not_found_err:
            raise RuntimeError(
                "Claude Code CLI not found. Please install it using: "
                "pip install claude-cli or follow installation instructions at "
                "https://docs.anthropic.com/en/docs/claude-code"
            ) from file_not_found_err
        except Exception as e:
            raise RuntimeError(f"Failed to execute Claude Code: {str(e)}") from e

    async def infer_batch_async(
        self,
        histories: list[list[dict[str, str]]],
        disable_system_prompt: bool = False,
        working_dirs: list[str] = None,
        **kwargs,
    ) -> list[str]:
        """
        Execute multiple research tasks using Claude Code CLI.

        Note: Claude Code tasks are executed sequentially to avoid conflicts
        and resource issues, as each task may be quite resource-intensive.

        :param histories: List of conversation histories
        :param disable_system_prompt: Ignored
        :param working_dirs: Optional list of working directories (one per history)
        :param kwargs: Additional arguments
        :return: List of Claude Code responses
        """
        results = []

        for i, history in enumerate(histories):
            try:
                print(f"Executing Claude Code task {i + 1}/{len(histories)}...")
                # Get working directory for this task if provided
                working_dir = working_dirs[i] if working_dirs and i < len(working_dirs) else None
                result = await self.infer_single_async(
                    history, disable_system_prompt, working_dir=working_dir, **kwargs
                )
                results.append(result)
            except Exception as e:
                error_msg = f"Task {i + 1} failed: {str(e)}"
                print(f"WARNING: {error_msg}")
                results.append(error_msg)

        return results

    async def logprobs_single_async(
        self, dialogue: list[dict[str, str]], return_summed: bool = True
    ) -> float | list[float]:
        """
        Claude Code does not support logprob calculations.
        This method is required by the Policy interface but will always raise an error.

        :param dialogue: OpenAI-format dialogue
        :return: Never returns, always raises NotImplementedError
        """
        raise NotImplementedError(
            "Claude Code does not support logprob calculations. "
            "Use a LocalModel or API model with logprob support for forecasting tasks."
        )
