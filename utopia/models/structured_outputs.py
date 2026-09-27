"""Dependency-free request fields shared by production and its schema probe.

This file can be loaded directly with importlib.util on the controller's Python
without importing the utopia package, Torch, vLLM, or a model client.
"""


def build_vllm_extra_body(schema=None, *, enable_thinking=True):
    """Preserve existing optional-field semantics using vLLM 0.12's schema API.

    The schema is passed through unchanged. A missing or empty schema adds no
    constraint. False/None thinking omits the chat-template override entirely;
    it does not send enable_thinking=False or alter the server's defaults.
    """
    extra_body = {}
    if schema:
        extra_body["structured_outputs"] = {"json": schema}
    if enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    return extra_body
