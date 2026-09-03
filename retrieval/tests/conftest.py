def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: end-to-end tests that load the bge-m3 model and build a LanceDB "
        "index (needs torch/FlagEmbedding/lancedb; ~2.3GB one-time download).",
    )
