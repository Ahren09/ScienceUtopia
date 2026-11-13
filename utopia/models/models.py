import base64
import json
import logging
import os
import re
import time
import threading
from pathlib import Path
from io import BytesIO
from typing import List, Optional, Union, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import google.generativeai as genai
import numpy as np
from PIL import Image
from openai import AzureOpenAI, OpenAI
from utopia.constants import IMPORTANT_NOTES
from utopia.config import SIMULATION_CONFIG
from utopia.models.structured_outputs import build_vllm_extra_body

logger = logging.getLogger(__name__)

class ExperimentTimeLimit(BaseException):
    """Stop new requests at the campaign deadline while retaining saved replies."""


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

    Also strips Qwen3 </think> content if present. Lone Unicode surrogates
    decoded from model JSON become U+FFFD so later prompts remain encodable;
    the supplied raw response text is unchanged.

    Args:
        text: Raw model response text

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If JSON parsing fails
    """
    def decode_json(candidate):
        value = json.loads(candidate)
        serialized = json.dumps(value, ensure_ascii=False)
        try:
            serialized.encode('utf-8')
        except UnicodeEncodeError:
            value = json.loads(
                serialized.encode('utf-16-le', errors='surrogatepass')
                .decode('utf-16-le', errors='replace'))
        return value

    text = text.strip()

    # For Qwen3 thinking mode, extract content after </think> tag
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Strategy 1: Direct parse
    try:
        return decode_json(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(json_block_pattern, text)
    for match in matches:
        try:
            return decode_json(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: Find JSON object with regex
    json_obj_pattern = r"\{[\s\S]*\}"
    match = re.search(json_obj_pattern, text)
    if match:
        try:
            return decode_json(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try fixing missing closing brace
    candidate = text
    json_match = re.search(r"\{[\s\S]*", text)
    if json_match:
        candidate = json_match.group()
    if candidate.count("{") - candidate.count("}") == 1:
        try:
            return decode_json(candidate + "}")
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


class ModelRequestError(BaseException):
    """Stop failed API batches after preserving every completed reply."""


def openai_schema(value):
    """Translate the existing agent schemas into the OpenAI strict-JSON subset."""
    if isinstance(value, list):
        return [openai_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: openai_schema(item) for key, item in value.items()
              if key not in ('default', 'const')}
    if 'const' in value:
        constant = value['const']
        result['enum'] = [constant]
        result.setdefault('type', 'integer' if isinstance(constant, int) else 'string')
    if result.get('type') == 'object' or 'properties' in result:
        properties = result.setdefault('properties', {})
        required = set(result.get('required', []))
        for key, schema in properties.items():
            if key not in required:
                properties[key] = {'anyOf': [schema, {'type': 'null'}]}
        result['required'] = list(properties)
        result['additionalProperties'] = False
    return result


class APIBatchModel(BaseLLM):
    """Shared ordered batches, bounded retries and request-ID continuation for HTTP models."""

    fail_on_error = False
    supports_response_formats = True

    def __init__(
        self, model_name, base_url, max_concurrent_requests=32, run_seed=None,
        response_cache_path=None, stop_at_utc=None, completion_token_floor=0,
        thinking_token_budget=None, api_key=None, sdk_retries=5,
        input_tokens_per_minute=0, output_tokens_per_minute=0, api_budget_usd=0,
    ):
        super().__init__(model_name)
        self.base_url = base_url
        self.max_concurrent_requests = max_concurrent_requests
        self.completion_token_floor = completion_token_floor
        self.thinking_token_budget = thinking_token_budget
        self.enable_thinking = SIMULATION_CONFIG['llm']['enable_thinking']
        self.run_seed = run_seed
        self.response_cache_path = Path(response_cache_path) if response_cache_path else None
        self.saved_responses = {}
        self.response_write_lock = threading.Lock()
        self.token_rates = [input_tokens_per_minute, output_tokens_per_minute]
        if min(self.token_rates) < 0:
            raise ValueError('Token rates must be nonnegative')
        if input_tokens_per_minute or api_budget_usd:
            import tiktoken
            self.input_tokenizer = tiktoken.get_encoding('o200k_base')
        self.token_balances = [0.0, 0.0]
        self.quota_time = time.monotonic()
        self.output_estimates = {}
        self.batch_cancelled = threading.Event()
        self.api_budget_usd = api_budget_usd
        self.cost_usd = self.pending_cost_usd = 0.0
        self.stop_at_timestamp = None
        if stop_at_utc:
            from datetime import datetime
            self.stop_at_timestamp = datetime.fromisoformat(
                stop_at_utc.replace('Z', '+00:00')).timestamp()
        if self.response_cache_path:
            self.response_cache_path.parent.mkdir(parents=True, exist_ok=True)
            if self.response_cache_path.exists():
                with self.response_cache_path.open() as stream:
                    for line in stream:
                        if line.strip():
                            row = json.loads(line)
                            if not row.get('cache_replay', False):
                                self.cost_usd += self.response_cost(row['response'].get('usage') or {})
                            if row.get('complete'):
                                self.saved_responses.setdefault(row['request_id'], []).append(row)
                                context = json.loads(row['request_id'])[0]
                                phase = str(context[0]) if context else 'default'
                                output = (row['response'].get('usage') or {}).get('completion_tokens', 0)
                                self.output_estimates[phase] = max(self.output_estimates.get(phase, 0), output)
        from utopia.models.request_audit import active_audit_for
        self.request_audit = active_audit_for(model_name)
        self.call_stats = dict(n_prompts=0, n_first_attempt_success=0, n_retries=0,
                               n_failures=0, elapsed_seconds=0.0,
                               prompt_tokens=0, completion_tokens=0)
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600.0,
                             max_retries=sdk_retries)
        logger.info(f"Initialized {type(self).__name__} at {base_url} (model: {model_name})")

    def build_payload(self, messages, temperature, max_tokens, response_format,
                      seed_ctx=None, item_index=0, attempt=0):
        raise NotImplementedError

    def response_details(self, response):
        choice = response.choices[0]
        usage = response.usage
        return (choice.message.content or '', choice.finish_reason,
                 (usage.prompt_tokens or 0, usage.completion_tokens or 0) if usage else (0, 0))

    def response_cost(self, usage):
        return 0.0

    def find_saved_response(self, request_id, payload):
        """Reuse matching request IDs; keep input checks when retry queues change order."""
        fields = ('model', 'messages', 'temperature')
        for row in reversed(self.saved_responses.get(request_id, [])):
            saved = row.get('payload', {})
            if all(saved.get(field) == payload.get(field) for field in fields):
                return row['response']
        return None

    def wait_for_quota(self, payload, phase):
        """Reserve input and predicted output; sleeping requests remain cancellable."""
        content = json.dumps([payload['messages'], payload.get('response_format')], ensure_ascii=False)
        input_tokens = 0
        if self.token_rates[0] or self.api_budget_usd:
            try:
                input_tokens = len(self.input_tokenizer.encode(content, disallowed_special=()))
            except ValueError as error:
                if not str(error).startswith('Regex error while tokenizing:'):
                    raise
                # Long whitespace runs can exceed the tokenizer's regex stack.
                # Split only the quota estimate; the request keeps its full text.
                input_tokens = sum(len(self.input_tokenizer.encode(
                    content[start:start + 4096], disallowed_special=()))
                    for start in range(0, len(content), 4096))
            input_tokens += 64
        while not self.batch_cancelled.is_set():
            with self.response_write_lock:
                now = time.monotonic()
                reserved = [input_tokens, self.output_estimates.get(phase, 1024)]
                delay = 0.0
                for index, rate in enumerate(self.token_rates):
                    if rate:
                        self.token_balances[index] = min(
                            max(rate, reserved[index]),
                            self.token_balances[index] + (now - self.quota_time) * rate / 60)
                        delay = max(delay, (reserved[index] - self.token_balances[index]) * 60 / rate)
                self.quota_time = now
                if delay <= 0:
                    for index, rate in enumerate(self.token_rates):
                        if rate:
                            self.token_balances[index] -= reserved[index]
                    return reserved
            time.sleep(min(delay, 1.0))
        raise ModelRequestError('Batch stopped; completed replies retained')

    def create_completion(self, *, seed_ctx=None, item_index=0, attempt=0, **payload):
        request_id = json.dumps([seed_ctx, item_index, attempt],
                                ensure_ascii=True, separators=(',', ':'))
        cache_path = getattr(self, 'response_cache_path', None)
        if cache_path and seed_ctx is not None:
            saved = self.find_saved_response(request_id, payload)
            if saved is not None:
                from openai.types.chat import ChatCompletion
                return ChatCompletion.model_validate(saved)
        phase = str(seed_ctx[0]) if seed_ctx else 'default'
        reserved = self.wait_for_quota(payload, phase)
        cutoff = getattr(self, 'stop_at_timestamp', None)
        if cutoff is not None and time.time() >= cutoff:
            raise ExperimentTimeLimit('Campaign compute cutoff reached; saved replies retained')
        reserved_cost = self.response_cost({
            'prompt_tokens': reserved[0],
            'completion_tokens': payload.get('max_completion_tokens', payload.get('max_tokens', 0)),
            'prompt_tokens_details': {'cache_write_tokens': reserved[0]},
        }) if self.api_budget_usd else 0.0
        with self.response_write_lock:
            if self.api_budget_usd and self.cost_usd + self.pending_cost_usd + reserved_cost > self.api_budget_usd:
                self.batch_cancelled.set()
                raise ModelRequestError(f'API budget reached (${self.cost_usd:.2f}); saved replies retained')
            self.pending_cost_usd += reserved_cost
        call = self.client.chat.completions.create
        started_ns = time.time_ns()
        try:
            if self.request_audit is None:
                response = call(**payload)
            else:
                response = self.request_audit.create(
                    call, seed_ctx=seed_ctx, item_index=item_index, attempt=attempt, **payload)
        except Exception as error:
            with self.response_write_lock:
                self.pending_cost_usd -= reserved_cost
            if getattr(error, 'status_code', None) in (400, 401, 403, 404):
                self.batch_cancelled.set()
                raise ModelRequestError(f'Model access failed: {error}') from error
            raise
        _, _, actual = self.response_details(response)
        with self.response_write_lock:
            self.pending_cost_usd -= reserved_cost
            self.cost_usd += self.response_cost(response.usage.model_dump() if response.usage else {})
            self.output_estimates[phase] = max(self.output_estimates.get(phase, 0), actual[1])
            for index, rate in enumerate(self.token_rates):
                if rate:
                    self.token_balances[index] = min(
                        rate, self.token_balances[index] + reserved[index] - actual[index])
        if cache_path:
            text, finish, _ = self.response_details(response)
            complete = finish == 'stop'
            if complete:
                try:
                    result = parse_json_response(text)
                    fmt = payload.get('response_format', {}).get('json_schema', {})
                    if fmt.get('schema'):
                        from jsonschema import validate
                        validate(result, fmt['schema'])
                except Exception:
                    complete = False
            record = {
                'request_id': request_id, 'time_ns': time.time_ns(),
                'started_ns': started_ns, 'elapsed_seconds': (time.time_ns() - started_ns) / 1e9,
                'concurrency': self.max_concurrent_requests,
                'cache_replay': False, 'complete': complete,
                'payload': payload, 'response': response.model_dump(mode='json'),
            }
            serialized = json.dumps(record, ensure_ascii=True)
            with self.response_write_lock:
                with cache_path.open('a') as stream:
                    stream.write(serialized + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                if complete and seed_ctx is not None:
                    self.saved_responses.setdefault(request_id, []).append(json.loads(serialized))
        return response

    def generate(self, prompt=None, image=None, temperature=0.7,
                 system_prompt="This is a simulation of an academic ecosystem, where researchers choose research directions and submit papers, reviewers conduct peer reviews of papers, and funding agencies allocate fundings.",
                 response_format=None, message_history=None, max_tokens=2048, seed_ctx=None):
        """Generate one structured reply with the same provider transport as batches."""
        if image is not None:
            logger.warning('Image inputs are ignored by the simulation HTTP backend')
        messages = list(message_history or [])
        if not messages and system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        payload = self.build_payload(messages, temperature, max(max_tokens, self.completion_token_floor),
                                     response_format, seed_ctx)
        response = self.create_completion(seed_ctx=seed_ctx, **payload)
        text, finish, _ = self.response_details(response)
        if finish != 'stop':
            raise ValueError(f'Incomplete model response: {finish}')
        result = parse_json_response(text.strip())
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': text}]})
        return result, messages

    def generate_batch(self, prompts, response_format=None, temperature=0.7,
                       system_prompt="This is a simulation of an academic ecosystem, where researchers choose research directions and submit papers, reviewers conduct peer reviews of papers, and funding agencies allocate fundings.",
                       max_tokens=SIMULATION_CONFIG['llm']['max_tokens'], desc=None,
                       seed_ctx=None, item_ids=None, on_result=None, response_formats=None):
        """Execute one batch up to the configured concurrency; return in input order."""
        if not prompts:
            return []
        self.batch_cancelled.clear()
        max_tokens = max(max_tokens, self.completion_token_floor)
        if item_ids is not None and (len(item_ids) != len(prompts) or len(set(item_ids)) != len(item_ids)):
            raise ValueError('Each prompt requires one unique item ID')
        if response_formats is not None and len(response_formats) != len(prompts):
            raise ValueError('Each prompt requires one response format')
        formats = response_formats if response_formats is not None else [response_format] * len(prompts)

        def call_single(index):
            messages = [{'role': 'system', 'content': system_prompt}] if system_prompt else []
            messages.append({'role': 'user', 'content': f'{IMPORTANT_NOTES}\n\n{prompts[index]}'})
            context = (*(seed_ctx or ()), item_ids[index]) if item_ids is not None else seed_ctx
            position = 0 if item_ids is not None else index
            tokens = [0, 0]
            finish, error_message, retries, cache_replay = None, None, 0, False
            started = time.perf_counter()
            attempts = range(3)
            for saved_attempt in range(3):
                saved_payload = self.build_payload(messages, temperature, max_tokens, formats[index],
                                                   context, position, saved_attempt)
                saved_id = json.dumps([context, position, saved_attempt], ensure_ascii=True, separators=(',', ':'))
                if self.find_saved_response(saved_id, saved_payload) is not None:
                    attempts = [saved_attempt]
                    break
            for attempt in attempts:
                try:
                    payload = self.build_payload(messages, temperature, max_tokens, formats[index],
                                                 context, position, attempt)
                    request_id = json.dumps([context, position, attempt],
                                            ensure_ascii=True, separators=(',', ':'))
                    cache_replay = self.find_saved_response(request_id, payload) is not None
                    response = self.create_completion(seed_ctx=context, item_index=position,
                                                      attempt=attempt, **payload)
                    text, finish, usage = self.response_details(response)
                    if not cache_replay:
                        tokens[0] += usage[0]
                        tokens[1] += usage[1]
                    if finish != 'stop':
                        raise ValueError(f'Incomplete model response: {finish} (limit {max_tokens})')
                    result = parse_json_response(text.strip())
                    if result is None and formats[index] is not None:
                        raise ValueError('Structured output is empty')
                    if payload.get('response_format', {}).get('json_schema', {}).get('schema'):
                        from jsonschema import validate
                        validate(result, payload['response_format']['json_schema']['schema'])
                    messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': text}]})
                    return index, (result, messages), dict(success=True, n_retries=retries,
                        prompt_tokens=tokens[0], completion_tokens=tokens[1], cache_replay=cache_replay,
                        finish_reason=finish, max_tokens=max_tokens,
                        elapsed_seconds=time.perf_counter() - started, time_ns=time.time_ns())
                except Exception as error:
                    retries += 1
                    error_message = str(error)
                    logger.warning(f'Prompt {index} attempt {attempt + 1}/3 failed: {error}')
                    # Drain the batch before the caller selects a lower concurrency.
                    if self.fail_on_error and (getattr(error, 'status_code', None) == 429
                                              or type(error).__name__ == 'APITimeoutError'):
                        break
            return index, (None, messages), dict(success=False, n_retries=retries,
                prompt_tokens=tokens[0], completion_tokens=tokens[1], cache_replay=cache_replay,
                finish_reason=finish, max_tokens=max_tokens, error=error_message,
                elapsed_seconds=time.perf_counter() - started, time_ns=time.time_ns())

        label = desc or f'batch of {len(prompts)} prompts'
        logger.info(f'[{type(self).__name__}] Generating {label}; concurrency={self.max_concurrent_requests}')
        started = time.perf_counter()
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            futures = [executor.submit(call_single, index) for index in range(len(prompts))]
            try:
                for future in tqdm(as_completed(futures), total=len(prompts), desc=label):
                    index, result, info = future.result()
                    results[index] = result
                    self.call_stats['n_prompts'] += 1
                    for key in ('n_retries', 'prompt_tokens', 'completion_tokens'):
                        self.call_stats[key] += info[key]
                    if info['success'] and info['n_retries'] == 0:
                        self.call_stats['n_first_attempt_success'] += 1
                    if not info['success']:
                        self.call_stats['n_failures'] += 1
                    if on_result is not None:
                        on_result(index, result, info)
                    if self.fail_on_error and not info['success']:
                        raise ModelRequestError(
                            f'API request failed: {info["error"]}; completed replies are saved.')
            except BaseException:
                self.batch_cancelled.set()
                for future in futures:
                    future.cancel()
                raise
        self.call_stats['elapsed_seconds'] += time.perf_counter() - started
        return results


class VLLMServerModel(APIBatchModel):
    """OpenAI-compatible chat transport for the existing vLLM services."""

    def __init__(self, model_name, base_url='http://localhost:8000/v1', **kwargs):
        super().__init__(model_name, base_url, api_key=os.environ.get('VLLM_API_KEY', 'EMPTY'), **kwargs)
        self.client.models.list()

    def request_seed(self, seed_ctx, index, attempt):
        if self.run_seed is None:
            return None
        from utopia.utils.seeding import derive_seed
        return derive_seed(self.run_seed, *(seed_ctx or ('noctx',)), index, attempt)

    def build_extra_body(self, response_format=None):
        schema = (response_format.get('json_object', {}).get('schema')
                  if response_format and response_format.get('type') == 'json_schema' else None)
        extra = build_vllm_extra_body(schema, enable_thinking=self.enable_thinking)
        if getattr(self, 'thinking_token_budget', None) is not None:
            extra['thinking_token_budget'] = self.thinking_token_budget
        return extra

    def build_payload(self, messages, temperature, max_tokens, response_format,
                      seed_ctx=None, item_index=0, attempt=0):
        extra = self.build_extra_body(response_format)
        seed = self.request_seed(seed_ctx, item_index, attempt)
        if seed is not None:
            extra['seed'] = seed
        return dict(model=self.model_name, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, extra_body=extra or None)


class BedrockModel(APIBatchModel):
    """Bedrock OpenAI chat transport with online concurrent inference."""

    fail_on_error = True

    def __init__(self, model_name, base_url=None, **kwargs):
        if kwargs.get('api_budget_usd') and model_name != 'openai.gpt-5.6-luna':
            raise ValueError('API budget pricing is currently defined for openai.gpt-5.6-luna')
        region = os.environ.get('AWS_REGION', 'us-west-2')
        base_url = base_url or f'https://bedrock-mantle.{region}.api.aws/openai/v1'
        key = os.environ.get('AWS_BEARER_TOKEN_BEDROCK')
        if not key:
            raise ModelRequestError('AWS_BEARER_TOKEN_BEDROCK is required')
        super().__init__(model_name, base_url, api_key=key, sdk_retries=0, **kwargs)

    def response_cost(self, usage):
        if self.model_name != 'openai.gpt-5.6-luna':
            return 0.0
        prompt = usage.get('prompt_tokens', 0)
        output = usage.get('completion_tokens', 0)
        details = usage.get('prompt_tokens_details') or {}
        write = details.get('cache_write_tokens', 0) or 0
        read = details.get('cached_tokens', 0) or 0
        prices = (.44, .55, .044, 1.98) if prompt > 272000 else (.22, .275, .022, 1.32)
        return (max(0, prompt-write-read)*prices[0] + write*prices[1]
                + read*prices[2] + output*prices[3]) / 1e6

    def build_payload(self, messages, temperature, max_tokens, response_format,
                      seed_ctx=None, item_index=0, attempt=0):
        fmt = {'type': 'json_object'}
        if response_format:
            source = response_format.get('json_object', response_format.get('json_schema', {}))
            fmt = dict(type='json_schema', json_schema=dict(
                name=source.get('name', 'response'), strict=True,
                schema=openai_schema(source['schema'])))
        payload = dict(model=self.model_name, messages=messages, temperature=temperature,
                       max_completion_tokens=max_tokens, reasoning_effort='none', response_format=fmt)
        if seed_ctx and seed_ctx[0] in ('phase3_reviews', 'keywords'):
            properties = fmt.get('json_schema', {}).get('schema', {}).get('properties', {})
            text_schema = (properties.get('justification', {})
                           if seed_ctx[0] == 'phase3_reviews'
                           else properties.get('keywords', {}).get('items', {}))
            saved_ids = [json.dumps([seed_ctx, item_index, number], ensure_ascii=True,
                                    separators=(',', ':')) for number in range(3)]
            # Keep saved replies reusable, including those from the older schema.
            if text_schema.get('type') == 'string' and not any(
                    self.find_saved_response(saved_id, payload) is not None
                    for saved_id in saved_ids):
                # Repeated escaped control characters can exhaust the output cap.
                text_schema['pattern'] = (r'^[^\u0000-\u001f\u007f-\u009f]*$'
                                          if seed_ctx[0] == 'phase3_reviews'
                                          else r'^[^\u001b]*$')
        return payload


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
