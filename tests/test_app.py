from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    def test_invalid_rtsp_address_is_rejected_before_opening_source(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        dashboard = AppTest.from_file(str(app_path)).run()
        self.assertFalse(dashboard.exception)

        dashboard.selectbox[0].select("Kamera IP (RTSP)").run()
        dashboard.text_input[0].set_value("https://camera.invalid/stream").run()
        dashboard.button[0].click().run()

        self.assertFalse(dashboard.exception)
        self.assertIn("Podaj adres", dashboard.error[0].value)
        with self.assertRaises(KeyError):
            dashboard.session_state["analysis_job"]
