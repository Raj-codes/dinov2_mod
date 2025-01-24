from setuptools import setup, find_packages

setup(
    name="dinov2_mod",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],  # Dependencies moved to requirements.txt
    author="RAI",
    description="A custom pipeline for feature extraction and prediction using DINOv2.",
    url="https://github.com/yourusername/dinov2_mod",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="open",  
    python_requires='>=3.7',
)
