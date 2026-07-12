"""Compatibility shim for old setuptools (stock JetPack / Python 3.6).

Modern toolchains build straight from ``pyproject.toml`` and ignore this file.
The Python 3.6 setuptools on the stock Jetson Nano image is too old to read
project metadata from ``pyproject.toml`` alone -- without this shim, an editable
install fails ("File 'setup.py' not found") or registers a blank ``UNKNOWN``
package. The metadata here mirrors ``pyproject.toml`` for that toolchain only;
``pyproject.toml`` remains the source of truth everywhere else.
"""

from setuptools import setup, find_packages

setup(
    name="duckiebot-lab",
    version="0.4.0",
    description="Application layer for the pyhut Duckiebot: experiments, "
                "motion primitives, demos.",
    author="Jes Fink-Jensen",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    # pyhut is the one-way dependency; it isn't on PyPI. On the robot, install it
    # first (pip install -e ../pyhut). Left out of install_requires here so the
    # old-setuptools path doesn't try to resolve a git URL it can't handle.
    extras_require={
        "test": ["pytest<7.1"],   # last line with solid Python 3.6 support
        "plot": ["matplotlib"],   # off-robot only, for the optional PNG
    },
    entry_points={
        "console_scripts": [
            "duckiebot-emi = duckiebot_lab.experiments.motor_emi:main",
            "duckiebot-rotate = duckiebot_lab.motion.demo:main",
        ],
    },
)