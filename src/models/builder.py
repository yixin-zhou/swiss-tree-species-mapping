from src.models.multimodal import MultiModalTreeSpeciesModel


def build_model(cfg):
    """Build the published reference architecture with the configured RGB weights."""
    return MultiModalTreeSpeciesModel(cfg)
