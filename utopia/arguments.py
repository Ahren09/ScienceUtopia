import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Academic Society Ecosystem Simulation"
    )
    parser.add_argument(
        '--num_years',
        type=int,
        default=4,
        help='Number of years to simulate (default: 5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory for results (default: ./simulation_output)'
    )
    
    

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--max_retries',
        type=int,
        default=5,
        help='Maximum number of retries for LLM calls (default: 5)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model to use for LLM (default: gpt-4o)'
    )

    parser.add_argument(
        '--industry_funding_mode',
        type=str,
        default='performance',
        choices=['performance', 'consistent'],
        help='Funding mode for industry researchers: "performance" (based on papers) or "consistent" (fixed allocation) (default: performance)'
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

    args = parser.parse_args()
    
    if args.debug:
        args.output_dir = os.path.join(args.output_dir, 'debug')
        
    args.visual_dir = os.path.join(args.output_dir, 'visual')
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.visual_dir, exist_ok=True)
    
    
    return args