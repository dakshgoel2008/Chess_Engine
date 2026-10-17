"""Setuptools build script registering the CFFI C extension module."""

from setuptools import setup

setup(
    cffi_modules=["castled/core/_build_engine.py:ffibuilder"],
)
