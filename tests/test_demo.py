"""Interactive smoke tests for the Streamlit research demonstration."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "demo" / "app.py"
EXPECTED_TABS = [
    "📊 Prediction & Conformal Interval",
    "🔍 Local Feature Attribution",
    "⚖️ Subgroup Error Audit",
]


def test_demo_renders_each_analysis_tab_without_exceptions():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run(timeout=60)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == EXPECTED_TABS

    # Inspect each tab's own output tree so one successful tab cannot mask
    # a missing or broken analysis panel in another tab.
    assert len(app.tabs[0].metric) == 3
    assert len(app.tabs[1].dataframe) == 1
    assert len(app.tabs[2].metric) == 2
    assert len(app.tabs[2].json) == 1


def test_demo_model_profile_and_feature_controls_can_be_changed():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run(timeout=60)
    assert not app.exception
    assert len(app.selectbox) == 1
    assert len(app.radio) == 1
    assert len(app.slider) == 10

    for model_name in ("Ridge", "Linear Regression", "Lasso", "Random Forest"):
        app.selectbox[0].select(model_name).run(timeout=60)
        assert app.selectbox[0].value == model_name
        assert not app.exception, f"Demo failed after selecting {model_name}"

    for profile in ("Illustrative Profile A", "Illustrative Profile B", "Benchmark Mean"):
        app.radio[0].set_value(profile).run(timeout=60)
        assert app.radio[0].value == profile
        assert not app.exception, f"Demo failed after selecting {profile}"

    app.slider[0].set_value(0.10).run(timeout=60)
    assert app.slider[0].value == pytest.approx(0.10)
    assert not app.exception, "Demo failed after changing a synthetic feature input"
