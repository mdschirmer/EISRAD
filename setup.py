import re
from setuptools import setup, find_packages

# Single source of truth for the version: eisrad/__init__.py
with open('eisrad/__init__.py') as f:
    version = re.search(r'__version__\s*=\s*"([^"]+)"', f.read()).group(1)

# Read the pinned dependencies directly from requirements.txt
with open('requirements.txt') as req:
    requirements = [line.strip() for line in req if line.strip()]

setup(
    name='eisrad',
    version=version,
    description='Evaluation of Image Segmentations using RADar plots',
    author='Markus D. Schirmer',
    author_email='software@markus-schirmer.com',
    url='https://github.com/mdschirmer/eisrad',
    packages=find_packages(),
    install_requires=requirements,  # Pinned versions from requirements.txt
    entry_points={
        'console_scripts': [
            'eisrad = eisrad.__main__:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
