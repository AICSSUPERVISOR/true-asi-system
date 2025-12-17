"""Tests for ASI Engine"""
import pytest
import asyncio
from src.core.asi_engine import ASIEngine

@pytest.mark.asyncio
async def test_asi_engine_initialization():
    """Test ASI engine initialization"""
    engine = ASIEngine()
    assert engine is not None
    assert engine.state.status == 'initializing'

@pytest.mark.asyncio
async def test_reasoning_engine():
    """Test reasoning engine"""
    engine = ASIEngine()
    result = await engine.reasoning_engine.reason("test query", {})
    assert 'combined_conclusion' in result
    assert result['confidence'] > 0

def test_asi_state():
    """Test ASI state"""
    engine = ASIEngine()
    state = engine.get_state()
    assert 'timestamp' in state
    assert 'status' in state
