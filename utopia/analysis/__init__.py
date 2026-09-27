"""Analysis tools. Importing the package does not load plotting or model libraries."""
__all__ = [
    "SimulationAnalyzer", "analyze_citations", "analyze_papers", "analyze_scores",
    "analyze_ecosystem", "generate_report",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import analyze_all_results
    return getattr(analyze_all_results, name)
