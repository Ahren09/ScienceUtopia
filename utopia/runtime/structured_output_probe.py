"""Manually check local vLLM structured output with the original small model."""

import argparse

from pydantic import BaseModel


class Answer(BaseModel):
    thought_process: str
    final_answer: float


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        gpu_memory_utilization=0.5,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=500,
        structured_outputs=StructuredOutputsParams(json=Answer.model_json_schema()),
    )
    prompt = "Calculate the square root of 144 and explain your step."
    outputs = llm.generate([prompt], sampling_params)
    for output in outputs:
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()
