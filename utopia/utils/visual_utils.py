import logging
import os

logger = logging.getLogger(__name__)

def load_visualization_data(args):
    from run_simulation import Simulation
    simulation = Simulation(
        llm=None,
        num_years=args.num_years,
        output_dir=args.output_dir,
        debug=args.debug,
        industry_funding_mode=args.industry_funding_mode,
        always_rerun=args.always_rerun,
        use_langchain=args.use_langchain
    )
    
    loaded_year = None

    for year in range(args.num_years, 0, -1):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_year_{year}.json")
        if os.path.exists(checkpoint_path):
            logger.info(f"Found checkpoint at year {year}, attempting to load...")
            if simulation.load_checkpoint(year):
                print(f"Loaded checkpoint for year {year}")
                loaded_year = year
                break
            
    return {
        "simulation": simulation, 
        "loaded_year": loaded_year
    }