import importlib

MODULES = [
    "src.system",
    "src.api.routes",
    "src.core.localization",
    "src.core.perception",
    "src.core.reconstruction",
    "src.core.security",
    "src.utils.helpers",
]

def test_imports():
    for module_name in MODULES:
        mod = importlib.import_module(module_name)
        assert mod is not None

def test_api_entry_exists():
    routes = importlib.import_module("src.api.routes")
    assert hasattr(routes, "app") or hasattr(routes, "router")

    