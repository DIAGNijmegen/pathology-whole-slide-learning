from setuptools import setup, find_packages

# Read the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="wsilearn",
    version="0.2.0",
    author="Witali Aswolinskiy",
    url="https://github.com/DIAGNijmegen/pathology-whole-slide-learning",
    packages=find_packages(),
    install_requires=requirements,  # Use the contents of requirements.txt
    long_description="Package for multiple instance learning/neural image compression with whole slide images.",
)
