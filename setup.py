from setuptools import setup


setup(
    name="namegen",
    version="0.1.0",
    description="Style-aware reusable fake-name generator",
    py_modules=["namegen"],
    entry_points={"console_scripts": ["namegen=namegen:main"]},
)
