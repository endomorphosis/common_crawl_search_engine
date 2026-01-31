from setuptools import find_packages, setup

setup(
    name="common-crawl-search-engine",
    version="0.1.0",
    description="Common Crawl indexing and search utilities.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
)
