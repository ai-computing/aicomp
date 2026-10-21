"""Shared pytest configuration.

opt_prime.accel targets transformers 5.x. On an older transformers, tests that need a model family or config
attribute it does not have are skipped instead of failing.
"""
import pytest

try:
    import transformers
    _TF_MAJOR = int(transformers.__version__.split(".")[0])
except Exception:                                   # transformers missing or unparsable version
    _TF_MAJOR = None

_TF4_GAPS = ("Unrecognized model identifier", "object has no attribute 'rope_parameters'",
             "does not recognize this architecture")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except (ValueError, AttributeError, KeyError) as exc:
        _reraise_as_skip(exc)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except (ValueError, AttributeError, KeyError) as exc:
        _reraise_as_skip(exc)


def _reraise_as_skip(exc):
    if _TF_MAJOR is not None and _TF_MAJOR < 5 and any(g in str(exc) for g in _TF4_GAPS):
        pytest.skip(f"needs transformers>=5 (installed {transformers.__version__}): {str(exc)[:80]}")
    raise exc
