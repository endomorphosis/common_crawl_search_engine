from pathlib import Path

from setuptools import find_packages, setup

root = Path(__file__).parent
readme = (root / "README.md").read_text(encoding="utf-8")

setup(
    name="common-crawl-search-engine",
    version="0.1.0",
    description="Common Crawl indexing and search utilities.",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "ccindex=common_crawl_search_engine.cli:main",
            "ccindex-mcp-server=common_crawl_search_engine.mcp_server:main",
            "ccindex-dashboard=common_crawl_search_engine.dashboard:main",
        ],
    },
)
