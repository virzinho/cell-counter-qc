from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_demo_and_quality_controls():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert int(app.metric[0].value) > 0
    app.slider[3].set_value(4.0).run()
    assert not app.exception
    assert int(app.metric[1].value) != 0
    app.selectbox(key="method").set_value("connected").run()
    assert not app.exception
    app.radio[0].set_value("Upload image").run()
    assert not app.exception
    assert app.info
