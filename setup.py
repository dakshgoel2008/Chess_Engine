from setuptools import setup

setup(
    cffi_modules=["castled/core/_build_engine.py:ffibuilder"],
)

