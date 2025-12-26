from setuptools import setup, find_packages

setup(
    name="pc-udrl",
    version="0.1.0",
    description="Offline UDRL with Pessimistic Command Generation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "pcudrl-hydra=main_hydra:main",
        ]
    },
)

