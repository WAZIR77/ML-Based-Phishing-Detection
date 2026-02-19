"""Deployment: predictor and Flask app."""
from .predictor import predict, predict_dict, load_artifacts

__all__ = ["predict", "predict_dict", "load_artifacts"]
