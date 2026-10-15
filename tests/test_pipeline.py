import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from cxr.pipeline import PipelineStopped, exclusive_project_run, run_stage


class PipelineLifecycleTests(unittest.TestCase):
    def test_project_lock_excludes_another_run_and_releases(self):
        with tempfile.TemporaryDirectory() as temp:
            fake_module = Path(temp) / "cxr" / "pipeline.py"
            with patch("cxr.pipeline.__file__", str(fake_module)):
                with exclusive_project_run(Path("first.json")):
                    with self.assertRaisesRegex(RuntimeError, "already running"):
                        with exclusive_project_run(Path("other.json")):
                            self.fail("A concurrent run acquired the GPU lock")
                with exclusive_project_run(Path("after.json")):
                    pass

    def test_stop_interrupts_a_silent_stage_and_reaps_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "fake_silent_stage.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
            stop = root / "STOP"
            children = []
            original_popen = subprocess.Popen

            def record_popen(*args, **kwargs):
                child = original_popen(*args, **kwargs)
                children.append(child)
                return child

            timer = threading.Timer(0.35, stop.touch)
            timer.start()
            try:
                env_path = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
                with patch.dict(os.environ, {"PYTHONPATH": env_path}), \
                     patch("cxr.pipeline.subprocess.Popen", side_effect=record_popen):
                    with self.assertRaises(PipelineStopped):
                        run_stage("predict", "fake_silent_stage", [], root, stop, lambda **values: None)
                self.assertTrue(children)
                self.assertTrue(all(child.poll() is not None for child in children))
            finally:
                timer.cancel()
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
