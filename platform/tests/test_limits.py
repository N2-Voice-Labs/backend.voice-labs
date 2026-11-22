"""Token-bucket memory bound."""

from src.platform.limits import TokenBucket


def test_full_buckets_are_evicted_once_the_table_is_large() -> None:
    bucket = TokenBucket(max_keys=5)
    for n in range(5):
        assert bucket.take(f"ip-{n}", 60, now=0.0) == 0.0
    assert len(bucket._state) == 5
    # 60/min refills one token per second: a minute later every old bucket is full again,
    # so a new key triggers a prune that drops them all.
    assert bucket.take("new", 60, now=120.0) == 0.0
    assert list(bucket._state) == ["new"]


def test_pruning_keeps_buckets_that_are_still_limiting() -> None:
    bucket = TokenBucket(max_keys=3)
    for _ in range(2):
        bucket.take("busy", 2, now=0.0)  # drained: next token only after 30 s
    for n in range(2):
        bucket.take(f"idle-{n}", 60, now=0.0)
    bucket.take("trigger", 60, now=10.0)  # prune: idle ones are full again, busy is not
    assert set(bucket._state) == {"busy", "trigger"}
    assert bucket.take("busy", 2, now=10.0) > 0  # still limited, not reset by the prune


def test_eviction_does_not_change_the_limit() -> None:
    bucket = TokenBucket(max_keys=1)
    assert [bucket.take("a", 2, now=0.0) for _ in range(3)][2] > 0
    bucket.take("b", 2, now=0.0)  # prune runs; "a" is not full yet so it survives
    assert bucket.take("a", 2, now=0.0) > 0


def test_table_does_not_grow_with_one_off_keys() -> None:
    bucket = TokenBucket(max_keys=100)
    for n in range(2000):
        bucket.take(f"ip-{n}", 6, now=n * 600.0)  # each key is idle for a full refill
    assert len(bucket._state) <= 100


def test_reset_clears_state() -> None:
    bucket = TokenBucket(max_keys=2)
    for n in range(5):
        bucket.take(f"k{n}", 1, now=0.0)
    bucket.reset()
    assert bucket._state == {} and bucket._prune_at == 2
