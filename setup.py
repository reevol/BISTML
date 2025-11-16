from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bistml",
    version="1.0.0",
    author="BIST AI Trading System Team",
    author_email="info@bistml.com",
    description="AI-driven quantitative trading system for Borsa Istanbul (BIST)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/BISTML",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=6.5.4",
            "ipython>=8.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bistml=src.ui.cli:main",
            "bistml-dashboard=run_dashboard:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
