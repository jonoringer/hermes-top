from setuptools import find_packages, setup


setup(
    name="hermes-top",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={"console_scripts": ["hermes-top=hermes_top.cli:main"]},
)
