"""
Batch Processing Utilities for LLM Calls

This module provides utilities for collecting and executing batched LLM calls
to improve efficiency when using vLLM or other batch-capable models.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single request to be processed in a batch

    Attributes:
        request_id: Unique identifier for this request (e.g., agent_id or paper_id)
        prompt: The prompt text to send to the LLM
        system_prompt: Optional system prompt override
        temperature: Sampling temperature for this request
        metadata: Optional metadata dict for tracking context
    """
    request_id: str
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class BatchProcessor:
    """Batch collection and execution utility for LLM calls

    This class provides a unified interface for batching LLM calls.
    It automatically detects if the underlying LLM supports batching
    and falls back to sequential processing if not.

    Example usage:
        processor = BatchProcessor(llm, max_batch_size=64)

        # Collect requests
        for agent in agents:
            processor.add_request(BatchRequest(
                request_id=agent.id,
                prompt=agent.get_prompt(),
                metadata={'agent_type': agent.type}
            ))

        # Execute batch
        results = processor.execute_batch()

        # Process results
        for request_id, (response_json, messages) in results.items():
            process_response(request_id, response_json)
    """

    def __init__(
        self,
        llm,
        max_batch_size: int = 64,
        default_system_prompt: Optional[str] = None
    ):
        """Initialize BatchProcessor

        Args:
            llm: LLM instance (GPT, VLLMModel, etc.)
            max_batch_size: Maximum number of requests per batch
            default_system_prompt: Default system prompt if not specified per-request
        """
        self.llm = llm
        self.max_batch_size = max_batch_size
        self.default_system_prompt = default_system_prompt
        self.pending_requests: List[BatchRequest] = []

    def supports_batching(self) -> bool:
        """Check if the underlying LLM supports batch generation

        Returns:
            True if LLM has generate_batch method, False otherwise
        """
        return hasattr(self.llm, 'generate_batch') and callable(getattr(self.llm, 'generate_batch'))

    def add_request(self, request: BatchRequest) -> None:
        """Add a request to the pending batch

        Args:
            request: BatchRequest to add
        """
        self.pending_requests.append(request)

    def clear(self) -> None:
        """Clear all pending requests"""
        self.pending_requests = []

    def get_pending_count(self) -> int:
        """Get number of pending requests

        Returns:
            Number of pending requests
        """
        return len(self.pending_requests)

    def execute_batch(
        self,
        clear_after: bool = True
    ) -> Dict[str, Tuple[Dict, List[Dict]]]:
        """Execute all pending requests as a batch

        If the LLM supports batching, processes all requests in a single call.
        Otherwise, falls back to sequential processing.

        Args:
            clear_after: If True, clear pending requests after execution

        Returns:
            Dict mapping request_id -> (response_json, message_history)
        """
        if not self.pending_requests:
            return {}

        results = {}

        if self.supports_batching():
            results = self._execute_batch_native()
        else:
            results = self._execute_sequential()

        if clear_after:
            self.clear()

        return results

    def _execute_batch_native(self) -> Dict[str, Tuple[Dict, List[Dict]]]:
        """Execute batch using native generate_batch method

        Returns:
            Dict mapping request_id -> (response_json, message_history)
        """
        results = {}

        # Process in chunks if exceeding max batch size
        for i in range(0, len(self.pending_requests), self.max_batch_size):
            chunk = self.pending_requests[i:i + self.max_batch_size]

            # Extract prompts and determine common system prompt
            prompts = [req.prompt for req in chunk]

            # Use first request's system prompt or default
            system_prompt = chunk[0].system_prompt or self.default_system_prompt

            # Use first request's temperature (could be made per-request in future)
            temperature = chunk[0].temperature

            logger.info(f"Executing batch chunk: {len(chunk)} requests")

            # Call batch generate
            batch_results = self.llm.generate_batch(
                prompts=prompts,
                temperature=temperature,
                system_prompt=system_prompt
            )

            # Map results back to request IDs
            for req, result in zip(chunk, batch_results):
                results[req.request_id] = result

        return results

    def _execute_sequential(self) -> Dict[str, Tuple[Dict, List[Dict]]]:
        """Execute requests sequentially (fallback for non-batch LLMs)

        Returns:
            Dict mapping request_id -> (response_json, message_history)
        """
        results = {}

        logger.info(f"Executing {len(self.pending_requests)} requests sequentially (no batch support)")

        for req in self.pending_requests:
            system_prompt = req.system_prompt or self.default_system_prompt

            response_json, messages = self.llm.generate(
                prompt=req.prompt,
                temperature=req.temperature,
                system_prompt=system_prompt
            )

            results[req.request_id] = (response_json, messages)

        return results
