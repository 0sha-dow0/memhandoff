"""Shared fixtures for storage integration tests."""

import pytest

from open_context.models import Session
from open_context.storage import Database, Repository


@pytest.fixture
def db():
    with Database() as database:
        yield database


@pytest.fixture
def repo(db):
    return Repository(db)


@pytest.fixture
def session(repo):
    return repo.add_session(Session(title="test session", source="fixtures"))
