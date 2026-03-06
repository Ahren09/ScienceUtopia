import argparse
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Academic Society Ecosystem Simulation"
    )
    parser.add_argument(
        '--num_years',
        type=int,
        default=10,
        help='Number of years to simulate (default: 10)'
    )
    parser.add_argument(
        '--start_year',
        type=int,
        default=2016,
        help='Starting real-world year for data partitioning (default: 2016). '
             'Each simulation year maps to a real-world year: Year 1 -> start_year, Year 2 -> start_year+1, etc.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results (default: ./simulation_output)'
    )
    
    

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debug mode'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Verbose mode'
    )
    
    parser.add_argument(
        '--max_retries',
        type=int,
        default=10,
        help='Maximum number of retries for LLM calls (default: 5)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model to use for LLM. OpenAI models (gpt-4o, gpt-4o-mini, etc.) use OpenAI API. '
             'Other models (e.g., Qwen/Qwen3-8B) use vLLM. (default: gpt-4o)'
    )

    # vLLM-specific arguments
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism when using vLLM (default: 1)'
    )
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.8,
        help='Fraction of GPU memory to use for vLLM (default: 0.8)'
    )
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=8192,
        help='Maximum sequence length for vLLM models (default: 8192)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Maximum batch size for vLLM batch generation (default: 64)'
    )
    parser.add_argument(
        '--vllm_url',
        type=str,
        default=None,
        help='URL of external vLLM server (e.g., http://localhost:8000/v1). '
             'Start server with: vllm serve <model> --gpu-memory-utilization 0.8'
    )

    parser.add_argument(
        '--industry_funding_mode',
        type=str,
        default='performance',
        choices=['performance', 'consistent'],
        help='Funding mode for industry researchers: "performance" (based on papers) or "consistent" (fixed allocation) (default: performance)'
    )

    parser.add_argument(
        '--funding_allocation_mode',
        type=str,
        default='fixed',
        choices=['fixed', 'consumption_based'],
        help='Funding allocation mode: "fixed" (config base budget per winner) or "consumption_based" (derived from consumption history) (default: consumption_based)'
    )

    parser.add_argument(
        '--always_rerun',
        action='store_true',
        default=False,
        help='Ignore all checkpoints and start simulation from scratch (default: resume from latest checkpoint if available)'
    )
    parser.add_argument(
        '--start_phase',
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4, 5],
        help='Start from specific phase (0=resubmissions, 1=directions, 2=submission, 3=review, 4=decisions, 5=funding)'
    )
    parser.add_argument(
        '--use_langchain',
        action='store_true',
        default=False,
        help='Use LangChain for RAG (by default, we use SentenceTransformer)'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='default',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--weighted_funding_assignment',
        action='store_true',
        help='Funding assignment is weighted by the maturity of the paper'
    )
    parser.add_argument(
        '--collaboration_mode',
        type=str,
        default='cross_institute',
        choices=['intra_institute', 'cross_institute'],
        help='Collaboration mode for preferential attachment experiment: '
             '"intra_institute" (same institution only) or "cross_institute" (different institutions only) '
             '(default: cross_institute)'
    )

    args = parser.parse_args()

    # Set default experiment name based on model name
    if args.experiment_name == 'default':
        model_short = args.model.split('/')[-1]
        args.experiment_name = f'default_{model_short}'


    if args.debug:
        args.output_dir = os.path.join(args.output_dir, 'debug', args.experiment_name)
        
    elif args.experiment_name:
        args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    
    else:
        raise ValueError("Please specify the `experiment_name` or enable debug mode.")
        
    args.visual_dir = os.path.join(args.output_dir, 'visual')
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.visual_dir, exist_ok=True)
    
    
    return args