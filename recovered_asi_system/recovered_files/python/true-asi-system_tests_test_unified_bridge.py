import pytest
from unified_bridge import get_bridge, StateOfTheArtBridge, ModelCapability

@pytest.fixture
def bridge():
    return get_bridge()

def test_bridge_initialization(bridge: StateOfTheArtBridge):
    assert bridge is not None
    assert len(bridge.model_registry) > 0

def test_model_selection(bridge: StateOfTheArtBridge):
    model = bridge.select_model("test", capability=ModelCapability.CODE_GENERATION)
    assert model is not None
    assert ModelCapability.CODE_GENERATION in model.capabilities

def test_generation(bridge: StateOfTheArtBridge):
    response = bridge.generate("test", "hello world")
    assert isinstance(response, str)
    assert len(response) > 0
