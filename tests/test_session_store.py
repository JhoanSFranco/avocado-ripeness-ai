from avocado_ai.services.session_store import SessionStore
from tests.conftest import make_prediction

P = make_prediction([0.1, 0.1, 0.6, 0.1, 0.1])


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_append_accumulates_per_session():
    store = SessionStore()
    assert len(store.append("a", P)) == 1
    assert len(store.append("a", P)) == 2
    assert len(store.append("b", P)) == 1


def test_clear():
    store = SessionStore()
    store.append("a", P)
    assert store.clear("a") is True
    assert store.clear("a") is False
    assert len(store.append("a", P)) == 1


def test_sessions_expire():
    clock = FakeClock()
    store = SessionStore(ttl_seconds=10, clock=clock)
    store.append("a", P)
    clock.now = 11
    assert len(store) == 0
    assert len(store.append("a", P)) == 1  # starts fresh


def test_least_recently_used_session_is_evicted():
    store = SessionStore(max_sessions=2)
    store.append("a", P)
    store.append("b", P)
    store.append("a", P)  # "a" becomes most recent
    store.append("c", P)  # evicts "b"
    assert store.clear("b") is False
    assert store.clear("a") is True


def test_images_per_session_are_capped():
    store = SessionStore(max_images=3)
    for _ in range(5):
        snapshot = store.append("a", P)
    assert len(snapshot) == 3
