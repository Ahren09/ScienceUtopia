import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import List, Optional, Union, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from PIL import Image
from openai import AzureOpenAI, OpenAI
from utopia.constants import IMPORTANT_NOTES
from utopia.config import SIMULATION_CONFIG
from utopia.models.structured_outputs import build_vllm_extra_body

logger = logging.getLogger(__name__)


class BaseLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class GPT(BaseLLM):

    def __init__(self, model_name):
        super().__init__(model_name)
        if os.getenv("AZURE_OPENAI_ENDPOINT", None) is not None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if azure_endpoint is None:
                raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
            self.client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                      api_version=os.getenv('OPENAI_API_VERSION'),
                                      azure_endpoint=azure_endpoint,
                                      azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
                                      )

        elif os.getenv("OPENAI_API_KEY", None) is not None:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError("No API key found")

    def generate(self, prompt: Optional[str] = None, image: Optional[List[Union[str, Image.Image]]] = None,
                 temperature: float = 0.7, system_prompt: Optional[
                str] = "This is a simulation of an academic ecosystem, where researchers choose research directions and submit papers, reviewers conduct peer reviews of papers, and funding agencies allocate fundings.",
                 response_format=None, message_history: Optional[List[Dict]] = []) -> Tuple[Dict, List[Dict]]:
        # System instructions for the GPT model

        # Handle string prompt with image formatting similar to process_sample_proprietary_models
        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        # Add images if provided
        if image is not None:
            if isinstance(image, Image.Image):
                image = [image]

            if isinstance(image, list):
                for img in image:
                    if isinstance(img, str):
                        encoded_image = load_base64_image(img)
                    else:
                        # Convert PIL Image to base64
                        img_buffer = BytesIO()

                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(img_buffer, format='JPEG')
                        encoded_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                    })

        messages = []
        if message_history != []:
            new_message = [{
                "role": "user",
                "content": content
            }]
            messages = message_history + new_message

        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]

        assert len(messages) > 0, "Message history is empty"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}  # response_format
        )

        response_content = response.choices[0].message.content

        if response_content is None:
            raise ValueError("Empty response from model")

        response_json = json.loads(response_content.strip())
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": response_content
            }]
        })

        return response_json, messages


class Gemini(BaseLLM):
    def __init__(self, model_name, **kwargs):
        import google.generativeai as genai

        super().__init__(model_name)
        self.model_name = model_name
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.temperature = kwargs.get('temperature', 0.0)

        # Check if Google API key is set
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini models")

        # Configure the API
        genai.configure(api_key=api_key)

        # Initialize the Gemini model
        self.client = genai.GenerativeModel(model_name)
        self.config = genai.types.GenerationConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature
        )

    def generate(self, image: Union[str, Image.Image, List[Union[str, Image.Image]]] = None, prompt: str = None,
                 temperature: float = 0.0, top_p: float = 0.7):
        import google.generativeai as genai

        # Update config if temperature is provided
        if temperature != self.temperature:
            self.config = genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=temperature
            )

        content_parts = []
        # Add text prompt first

        # Add images if provided
        if image is not None:
            if isinstance(image, list):
                for img in image:
                    if isinstance(img, str):
                        img = Image.open(img).convert('RGB')
                    content_parts.append(img)
            else:
                if isinstance(image, str):
                    image = Image.open(image).convert('RGB')
                content_parts.append(image)

        if prompt:
            content_parts.append(prompt)

        response = self.client.generate_content(content_parts, generation_config=self.config)
        response.resolve()
        return response.text.strip()

    def complete(self, prompt: str):
        """
        Generate text-only response (for compatibility with other models).
        
        Args:
            prompt: Text prompt
            
        Returns:
            Generated text response
        """
        return self.generate(prompt=prompt)

    def embed_query(self, text: str):
        """
        Generate embeddings for text (for compatibility with other models).
        Note: This is a placeholder implementation since Gemini doesn't have a direct embedding API.
        
        Args:
            text: Text to embed
            
        Returns:
            Placeholder embedding (zeros vector)
        """
        # For now, return a placeholder embedding since Gemini doesn't have a direct embedding API
        # This will cause the similarity computation to return random results
        return np.zeros(768)  # Return a zero vector as placeholder


def parse_json_response(text: str) -> Dict:
    """Parse JSON from model response text

    Attempts multiple parsing strategies:
    1. Direct json.loads()
    2. Extract from ```json ... ``` blocks
    3. Regex match for {...} object

    Also strips Qwen3 </think> content if present.

    Args:
        text: Raw model response text

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If JSON parsing fails
    """
    text = text.strip()

    # For Qwen3 thinking mode, extract content after </think> tag
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(json_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: Find JSON object with regex
    json_obj_pattern = r"\{[\s\S]*\}"
    match = re.search(json_obj_pattern, text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try fixing missing closing brace
    candidate = text
    json_match = re.search(r"\{[\s\S]*", text)
    if json_match:
        candidate = json_match.group()
    if candidate.count("{") - candidate.count("}") == 1:
        try:
            return json.loads(candidate + "}")
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON from response: {text[:500]}...")


class VLLMModel(BaseLLM):
    """vLLM-accelerated model wrapper with same interface as GPT class

    This class wraps vLLM for efficient batched inference while maintaining
    the same generate() interface as the GPT class for compatibility.
    """
    
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = None,
        distributed_executor_backend: str = "mp",
        **kwargs
    ):
        """Initialize vLLM model

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen3-8B")
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (optional, vLLM auto-detects if None)
            distributed_executor_backend: Backend for multi-GPU ("mp" for single, "nccl" for multi)
        """
        from vllm import LLM as vLLMLLM

        super().__init__(model_name)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enable_thinking = SIMULATION_CONFIG['llm']['enable_thinking']

        # Ensure vLLM uses 'spawn' method for CUDA compatibility
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

        # Build LLM initialization arguments
        llm_kwargs = {
            'model': model_name,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'trust_remote_code': True,
        }

        # Only add max_model_len if explicitly specified
        if max_model_len is not None:
            llm_kwargs['max_model_len'] = max_model_len

        # Only add distributed backend for multi-GPU
        if tensor_parallel_size > 1:
            llm_kwargs['distributed_executor_backend'] = distributed_executor_backend

        # Initialize vLLM engine
        logger.info(f"Initializing vLLM with model {model_name}")
        self.model = vLLMLLM(**llm_kwargs)

        # Get tokenizer from model (preferred method)
        self.tokenizer = self.model.get_tokenizer()

        logger.info(f"vLLM model {model_name} loaded successfully")

    def _format_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        message_history: Optional[List[Dict]] = None
    ) -> str:
        """Format messages using the model's chat template

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            message_history: Optional message history

        Returns:
            Formatted input text for the model
        """
        messages = []

        if message_history:
            messages = message_history.copy()
            messages.append({"role": "user", "content": prompt})
        else:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )

        return formatted

    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from model response text (delegates to module-level function)"""
        return parse_json_response(response_text)

    def generate(
        self,
        prompt: Optional[str] = None,
        image: Optional[List[Union[str, Image.Image]]] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = "This is a simulation of an academic ecosystem, where researchers choose research directions and submit papers, reviewers conduct peer reviews of papers, and funding agencies allocate fundings.",
        response_format=None,
        message_history: Optional[List[Dict]] = None,
        max_tokens: int = 2048
    ) -> Tuple[Dict, List[Dict]]:
        """Generate response for a single prompt

        Args:
            prompt: User prompt
            image: Not supported for vLLM text models (ignored)
            temperature: Sampling temperature
            system_prompt: System prompt for context
            response_format: Ignored (JSON parsing is handled internally)
            message_history: Optional message history
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (parsed_json_dict, message_history)
        """
        if image is not None:
            logger.warning("VLLMModel does not support images, ignoring image parameter")

        # Format input
        formatted_input = self._format_messages(prompt, system_prompt, message_history)

        # Set sampling parameters
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["```\n\n", "\n\n\n"],  # Stop at end of JSON block
        )

        # Generate
        outputs = self.model.generate([formatted_input], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()

        # Parse JSON
        response_json = self._parse_json_response(response_text)

        # Build message history
        if message_history:
            messages = message_history.copy()
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        })

        return response_json, messages

    def generate_batch(
        self,
        prompts: List[str],
        response_format: Optional[Dict] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = "This is a simulation of an academic ecosystem, where researchers choose research directions and submit papers, reviewers conduct peer reviews of papers, and funding agencies allocate fundings.",
        max_tokens: int = SIMULATION_CONFIG['llm']['max_tokens'],
        desc: Optional[str] = None,
        seed_ctx: Optional[tuple] = None,  # accepted for API parity with VLLMServerModel; unused in-process
    ) -> List[Tuple[Dict, List[Dict]]]:
        """Process ALL prompts in a single vLLM call - no for-loop over prompts

        This is the key batching method that processes all prompts at once
        using vLLM's internal batching capabilities for maximum efficiency.

        Args:
            prompts: List of user prompts to process
            response_format: Optional dict with 'json_schema' containing the schema for guided decoding
            temperature: Sampling temperature
            system_prompt: System prompt for all requests
            max_tokens: Maximum tokens per response

        Returns:
            List of (parsed_json_dict, message_history) tuples, one per prompt
        """
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        if not prompts:
            return []

        # Step 1: Format all prompts with chat template
        formatted_inputs = [
            self._format_messages(prompt, system_prompt, None)
            for prompt in prompts
        ]

        # Step 2: Extract JSON schema for structured outputs if provided
        structured_outputs = None
        if response_format:
            # Extract schema from response_format dict
            # Expected format: {'type': 'json_schema', 'json_object': {'schema': {...}}}
            if response_format.get('type') == 'json_schema':
                json_object = response_format.get('json_object', {})
                json_schema = json_object.get('schema')
                if json_schema:
                    structured_outputs = StructuredOutputsParams(json=json_schema)

        # Step 3: SINGLE vLLM call for all prompts at once
        sampling_params_kwargs = {
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        # Only add stop sequences if not using structured outputs (they can conflict)
        if structured_outputs is None:
            sampling_params_kwargs['stop'] = ["```\n\n", "\n\n\n"]
        else:
            sampling_params_kwargs['structured_outputs'] = structured_outputs

        sampling_params = SamplingParams(**sampling_params_kwargs)

        label = desc or f"batch of {len(prompts)} prompts"
        logger.info(f"[vLLM] Generating {label} ...")
        outputs = self.model.generate(formatted_inputs, sampling_params=sampling_params)

        # Step 3: Parse JSON from each response
        results = []
        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text.strip()

            try:
                response_json = self._parse_json_response(response_text)
            except Exception as e:
                logger.warning(f"Prompt {i} JSON parse failed: {e}")
                response_json = None

            # Build message history for this prompt
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompts[i]})
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}]
            })

            results.append((response_json, messages))

        logger.info(f"Batch generation complete: {len(results)} responses")
        return results


class VLLMServerModel(BaseLLM):
    """Connects to an external vLLM server via OpenAI-compatible API.

    This avoids in-process model loading (30-300s) by connecting to a
    persistent vLLM server started separately (e.g., `vllm serve Qwen/Qwen3-0.6B`).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        max_concurrent_requests: int = 32,
        run_seed: Optional[int] = None,
        response_cache_dir=None,
        checkpoint_binding=None,
        model_revision=None,
        max_tokens_override=None,
        sdk_max_retries: int = 5,
        request_timeout: float = 600.0,
        thinking_override: Optional[bool] = None,
    ):
        if thinking_override is not None and type(thinking_override) is not bool:
            raise ValueError("thinking_override must be None or a bool")
        super().__init__(model_name)
        self.thinking_override = thinking_override
        self.model_revision = model_revision
        self.max_tokens_override = max_tokens_override
        if max_tokens_override is not None and (type(max_tokens_override) is not int or max_tokens_override <= 0):
            raise ValueError("max_tokens_override must be a positive integer")
        self.response_cache = None
        if response_cache_dir is not None:
            from pathlib import Path
            from utopia.models.response_cache import CompletedResponseCache
            from utopia.runtime.checkpointing import PortableCheckpointError, validate_binding
            binding = validate_binding(checkpoint_binding)
            if (binding['model'] != model_name or binding['model_revision'] != model_revision
                    or binding['seed'] != run_seed):
                raise PortableCheckpointError("Model client differs from portable cache binding")
            if binding["science_config"].get("thinking_override") is not thinking_override:
                raise PortableCheckpointError("Thinking override differs from portable cache binding")
            if sdk_max_retries != 0:
                raise PortableCheckpointError("Portable calls require sdk_max_retries=0")
            self.response_cache = CompletedResponseCache(
                Path(response_cache_dir) / "responses.sqlite", binding=binding)
        self.base_url = base_url
        self.max_concurrent_requests = max_concurrent_requests
        self.enable_thinking = (thinking_override if thinking_override is not None
                                else SIMULATION_CONFIG['llm']['enable_thinking'])
        # Run seed: when set, every request gets a deterministic per-request seed
        # derived from (run_seed, seed_ctx..., item_index, attempt).
        self.run_seed = run_seed
        from utopia.models.request_audit import active_audit_for
        self.request_audit = active_audit_for(model_name)
        # Cumulative call statistics for the run manifest and fallback-rate gates
        self.call_stats = {
            'n_prompts': 0,
            'n_first_attempt_success': 0,
            'n_retries': 0,
            'n_failures': 0,
            'elapsed_seconds': 0.0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
        }

        # Local vLLM servers need no key ("EMPTY"); authenticated remote deployments
        # set VLLM_API_KEY. Generous timeout + retries tolerate autoscale cold starts.
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            timeout=request_timeout,
            max_retries=sdk_max_retries,
        )

        # Test connection (warning only, not fatal)
        self.client.models.list()
        logger.info(f"Connected to vLLM server at {base_url} (model: {model_name})")

    def _request_seed(self, seed_ctx: Optional[tuple], idx: int, attempt: int) -> Optional[int]:
        if self.run_seed is None:
            return None
        from utopia.utils.seeding import derive_seed
        return derive_seed(self.run_seed, *(seed_ctx or ('noctx',)), idx, attempt)
        
    def _build_extra_body(self, response_format: Optional[Dict] = None) -> dict:
        """Translate internal response_format to vLLM 0.12 structured outputs."""
        schema = None
        if response_format and response_format.get('type') == 'json_schema':
            json_object = response_format.get('json_object', {})
            schema = json_object.get('schema')
        extra = build_vllm_extra_body(schema, enable_thinking=self.enable_thinking)
        override = getattr(self, "thinking_override", None)
        if override is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": override}
        return extra

    def _create_completion(self, *, seed_ctx=None, item_index=0, attempt=0, **payload):
        audit = self.request_audit
        cache = getattr(self, 'response_cache', None)
        callback = self.client.chat.completions.create
        if cache is not None:
            raw_callback = callback
            def callback(**request):
                return cache.create(raw_callback, seed_ctx=seed_ctx,
                                    item_index=item_index, attempt=attempt, **request)
        if audit is None:
            response = callback(**payload)
        else:
            response = audit.create(callback, seed_ctx=seed_ctx,
                                    item_index=item_index, attempt=attempt, **payload)
        if cache is not None and cache.stop is not None:
            cache.stop.check()
        return response

    def generate(
        self,
        prompt: Optional[str] = None,
        image: Optional[List[Union[str, Image.Image]]] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = "This is a simulation of an academic ecosystem, where researchers choose research directions and submit papers, reviewers conduct peer reviews of papers, and funding agencies allocate fundings.",
        response_format=None,
        message_history: Optional[List[Dict]] = None,
        max_tokens: int = 2048,
        seed_ctx: Optional[tuple] = None,
    ) -> Tuple[Dict, List[Dict]]:
        """Generate response via the external vLLM server.

        Returns:
            Tuple of (parsed_json_dict, message_history)
        """
        if image is not None:
            logger.warning("VLLMServerModel does not support images, ignoring image parameter")

        # Build messages (server applies chat template)
        if message_history:
            messages = message_history.copy()
            messages.append({"role": "user", "content": prompt})
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        if getattr(self, 'max_tokens_override', None) is not None:
            max_tokens = self.max_tokens_override
        extra_body = self._build_extra_body(response_format)
        request_seed = self._request_seed(seed_ctx, 0, 0)
        if request_seed is not None:
            extra_body['seed'] = request_seed

        response = self._create_completion(
            seed_ctx=seed_ctx,
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body if extra_body else None,
        )

        try:
            response_text = response.choices[0].message.content.strip()
            response_json = parse_json_response(response_text)
            if response_json is None and getattr(self, 'response_cache', None) is not None:
                raise ValueError("Portable response parsed as None")
        except Exception as error:
            if getattr(self, 'response_cache', None) is not None:
                from utopia.runtime.checkpointing import PortableCheckpointError
                failure = PortableCheckpointError(f"Invalid portable response for {seed_ctx!r}: {error}")
                if self.response_cache.stop is not None:
                    self.response_cache.stop.fail(failure)
                raise failure from error
            raise

        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        })

        return response_json, messages

    def batch_completions(self, executor, requests, *, seed_ctx, attempt):
        """Only HTTP runs in workers; the caller owns all cache and audit writes."""
        from utopia.runtime.checkpointing import PortableStop
        cache, audit = getattr(self, 'response_cache', None), self.request_audit
        pending, replies = {}, {}
        failure = None
        try:
            for idx, payload in requests:
                ticket = audit.before(payload, seed_ctx=seed_ctx, item_index=idx,
                                      attempt=attempt) if audit else None
                try:
                    key, payload_sha, saved = (cache.lookup(
                        seed_ctx=seed_ctx, item_index=idx, attempt=attempt, **payload)
                        if cache else (None, None, None))
                    if saved is None:
                        future = executor.submit(self.client.chat.completions.create, **payload)
                        pending[future] = (idx, ticket, key, payload_sha)
                except BaseException as error:
                    if audit:
                        audit.transport_error(ticket, error)
                    raise
                if saved is not None:
                    if audit:
                        audit.response(ticket, saved)
                    replies[idx] = saved
        except BaseException as error:
            failure = error
        # Drain every submitted request even after a failure, saving each reply.
        for future in as_completed(pending):
            idx, ticket, key, payload_sha = pending[future]
            try:
                try:
                    response = future.result()
                except Exception as error:
                    if cache:
                        if cache.stop is not None:
                            cache.stop.request_stop()
                        raise PortableStop(
                            f"LLM transport interrupted: {type(error).__name__}: {error}") from error
                    raise
                if cache:
                    response = cache.store(key, payload_sha, response)
            except BaseException as error:
                if isinstance(error, Exception) and cache is None:
                    replies[idx] = error
                else:
                    failure = failure or error
                try:
                    if audit:
                        audit.transport_error(ticket, error)
                except BaseException as audit_error:
                    failure = failure or audit_error
                continue
            try:
                if audit:
                    audit.response(ticket, response)
                replies[idx] = response
            except BaseException as error:
                failure = failure or error
        if failure is not None:
            raise failure
        if cache is not None and cache.stop is not None:
            cache.stop.check()
        return replies

    def generate_batch(
        self,
        prompts: List[str],
        response_format: Optional[Dict] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = "This is a simulation of an academic ecosystem, where researchers choose research directions and submit papers, reviewers conduct peer reviews of papers, and funding agencies allocate fundings.",
        max_tokens: int = SIMULATION_CONFIG['llm']['max_tokens'],
        desc: Optional[str] = None,
        seed_ctx: Optional[tuple] = None,
    ) -> List[Tuple[Dict, List[Dict]]]:
        """Small HTTP batches with ordered results and single-thread persistence."""
        if not prompts:
            return []
        if getattr(self, 'max_tokens_override', None) is not None:
            max_tokens = self.max_tokens_override
        base_extra_body = self._build_extra_body(response_format)
        results = [None] * len(prompts)
        label = desc or f"batch of {len(prompts)} prompts"
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor, \
                tqdm(total=len(prompts), desc=label) as progress:
            for start in range(0, len(prompts), self.max_concurrent_requests):
                remaining = list(range(start, min(len(prompts), start + self.max_concurrent_requests)))
                for attempt in range(3):
                    requests, messages = [], {}
                    for idx in remaining:
                        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                        msgs.append({"role": "user", "content": f"{IMPORTANT_NOTES}\n\n{prompts[idx]}"})
                        messages[idx] = msgs
                        extra = dict(base_extra_body)
                        seed = self._request_seed(seed_ctx, idx, attempt)
                        if seed is not None:
                            extra['seed'] = seed
                        requests.append((idx, dict(model=self.model_name, messages=msgs,
                            temperature=temperature, max_tokens=max_tokens,
                            extra_body=extra if extra else None)))
                    replies = self.batch_completions(
                        executor, requests, seed_ctx=seed_ctx, attempt=attempt)
                    retry = []
                    for idx in remaining:
                        try:
                            response = replies[idx]
                            if isinstance(response, Exception):
                                raise response
                            text = response.choices[0].message.content.strip()
                            parsed = parse_json_response(text)
                            if parsed is None and (response_format is not None or getattr(self, 'response_cache', None) is not None):
                                raise ValueError("Parsed JSON is None but structured output was expected")
                        except Exception as error:
                            self.call_stats['n_retries'] += 1
                            if attempt < 2:
                                logger.warning(f"Prompt {idx} attempt {attempt + 1}/3 failed: {error}. Retrying...")
                                retry.append(idx)
                                continue
                            if getattr(self, 'response_cache', None) is not None:
                                from utopia.runtime.checkpointing import PortableCheckpointError
                                failure = PortableCheckpointError(
                                    f"Portable request {seed_ctx!r}/{idx} exhausted 3 response attempts")
                                if self.response_cache.stop is not None:
                                    self.response_cache.stop.fail(failure)
                                raise failure from error
                            results[idx] = (None, messages[idx])
                            self.call_stats['n_failures'] += 1
                        else:
                            messages[idx].append({"role": "assistant", "content": [
                                {"type": "text", "text": text}]})
                            results[idx] = (parsed, messages[idx])
                            if getattr(response, 'usage', None):
                                self.call_stats['prompt_tokens'] += response.usage.prompt_tokens or 0
                                self.call_stats['completion_tokens'] += response.usage.completion_tokens or 0
                            self.call_stats['n_first_attempt_success'] += int(attempt == 0)
                        self.call_stats['n_prompts'] += 1
                        progress.update(1)
                    remaining = retry
                    if not remaining:
                        break
        self.call_stats['elapsed_seconds'] += time.time() - start_time
        logger.info(f"Batch generation complete: {len(results)} responses")
        return results


def load_base64_image(image_input: Union[str, Image.Image], format: str = "JPEG") -> str:
    """
    Encodes an image as a base64 string.

    This function can handle both file paths and PIL Image objects. For PIL Images,
    it converts them to RGB format if needed and saves them in the specified format.

    Args:
        image_input (Union[str, Image.Image]): Either a path to an image file or a PIL Image object.
        format (str): Image format for PIL Images (default: "JPEG"). Ignored for file paths.

    Returns:
        str: Base64-encoded string representation of the image.
    """
    if isinstance(image_input, str):
        # Handle file path - read directly as bytes
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image_input, Image.Image):
        # Handle PIL Image - convert to RGB if needed and encode
        img = image_input
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        buffer = BytesIO()
        img.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}. Expected str (file path) or PIL.Image")
