"""Smoke tests: the package installs and every layer module imports."""

import importlib

import pytest

import minq


def test_version():
    assert minq.__version__ == "0.0.1"


@pytest.mark.parametrize(
    "module", ["minq.circuit", "minq.simulator", "minq.passes", "minq.qasm"]
)
def test_layer_modules_import(module):
    importlib.import_module(module)
