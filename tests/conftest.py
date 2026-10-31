import os

import pytest

from propmeasure import cli, registry
from propmeasure.db import connect, migrate
from propmeasure.seed import seed

SMALL = {"base": 12, "train": 9, "val": 6, "test": 9, "live": 6}


@pytest.fixture(scope="session")
def db_url():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set (the CI job and the README show how to start a test Postgres)")
    os.environ["DATABASE_URL"] = url
    return url


@pytest.fixture(scope="session")
def seeded(db_url):
    """A small stack: seeded, a base model in production, a fine-tuned candidate, benchmark runs."""
    with connect(db_url) as c:
        c.execute("DROP SCHEMA IF EXISTS pm CASCADE")
        migrate(c)
        seed(c, SMALL, workers=2, log=lambda m: None)
        registry._cache.clear()
        cli.cmd_train_base(c, steps=30)
        cli.cmd_evaluate(c, cli.BASE_VERSION, "v1", quiet=True)
        cli.cmd_finetune(c, steps=30)
    return db_url


@pytest.fixture()
def conn(seeded):
    with connect(seeded) as c:
        yield c
