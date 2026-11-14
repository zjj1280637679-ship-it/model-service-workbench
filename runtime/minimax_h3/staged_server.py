"""Compatibility API entry point; staging is chosen by the model container owner."""
from server import create_app


def build_app():
    return create_app()
