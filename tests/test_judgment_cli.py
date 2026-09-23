import tempfile
import unittest
from pathlib import Path


class JudgmentIOTests(unittest.TestCase):
    def test_json_boundary_rejects_duplicate_keys_and_nonfinite_values(self):
        from openjev.judgment_cli import read_json

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "request.json"
            for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_json(path)
            path.write_text('{"nested":[{"type":"noul"}]}')
            self.assertEqual(read_json(path), {"nested": [{"type": "noul"}]})
