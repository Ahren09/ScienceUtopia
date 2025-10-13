
export AZURE_OPENAI_API_KEY="5f22813ecf31430fa4370ee207b859ad"
export AZURE_OPENAI_ENDPOINT="https://gang-azureopenai-test.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
export OPENAI_API_VERSION="2024-10-21"
export S2_API_KEY="N9fmegjjkT8hNk4RdTO4K36WfO8uoP206gdL24rX"
conda init
conda activate slideagent
# python ai_scientist/perform_ideation_temp_free.py --workshop-file ai_scientist/ideas/slideagent.md --model gpt-4o --max-num-generations 20 --num-reflections 5 | tee -a ideation.log
python launch_scientist_bfts.py --load_ideas ai_scientist/ideas/slideagent.json --add_dataset_ref --model_writeup gpt-4o --model_citation gpt-4o --model_review gpt-4o --model_agg_plots gpt-4o --num_cite_rounds 7 | tee -a experiment.log
