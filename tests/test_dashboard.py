import unittest
from pathlib import Path
class DashboardTests(unittest.TestCase):
    def test_dashboard_syntax(self):
        compile(Path("app.py").read_text(encoding="utf-8"),"app.py","exec")
