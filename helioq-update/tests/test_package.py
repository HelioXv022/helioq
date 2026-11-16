"""Smoke tests: the package installs and every layer module imports."""

import importlib

import pytest

import helioq


def test_version():
    assert helioq.__version__ == "0.0.1"


@pytest.mark.parametrize(
    "module",
    [
        "helioq.circuit",
        "helioq.simulator",
        "helioq.passes",
        "helioq.qasm",
        "helioq.draw",
        "helioq.stabilizer",
        "helioq.qec",
    ],
)
def test_layer_modules_import(module):
    importlib.import_module(module)
