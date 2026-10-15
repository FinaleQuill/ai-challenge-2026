"""CPU-only tests for the safe CheXpert model adapter; no weights download."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torchvision import models

import cxr.medical as medical


class MedicalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def test_input_normalization_restores_xray_units(self) -> None:
        gray = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).reshape(1, 1, 1, 5)
        rgb = gray.repeat(1, 3, 1, 1)
        means = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        stds = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        actual = medical.InputNormalization()((rgb - means) / stds)
        torch.testing.assert_close(actual, (gray - 0.5) * 2048, atol=0.0002, rtol=0)
        self.assertEqual(tuple(actual.shape), (1, 1, 1, 5))
        with self.assertRaises(ValueError):
            medical.InputNormalization()(torch.zeros(1, 1, 32, 32))

    def test_untrained_structure_roundtrips_without_source_checkpoint(self) -> None:
        with patch.object(medical, "_read_pretrained", side_effect=AssertionError("Unexpected file read")):
            model = medical.build_chexpert_densenet(pretrained=False, dropout=0.2).eval()
            self.assertEqual(tuple(model.features.conv0[1].weight.shape), (64, 1, 7, 7))
            self.assertEqual(tuple(model.classifier[-1].weight.shape), (10, 1024))
            images = torch.randn(2, 3, 64, 64)
            with torch.inference_mode():
                expected = model(images)
            self.assertEqual(tuple(expected.shape), (2, 10))
            buffer = io.BytesIO()
            torch.save({"state_dict": model.state_dict(), "config": {"dropout": 0.2}}, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
            restored = medical.build_chexpert_densenet(pretrained=False, dropout=0.2).eval()
            restored.load_state_dict(checkpoint["state_dict"], strict=True)
            with torch.inference_mode():
                self.assertTrue(torch.equal(expected, restored(images)))

    def test_verified_source_rows_are_copied_and_no_finding_is_new(self) -> None:
        source = models.densenet121(weights=None, num_classes=18)
        source.features.conv0 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            for row in range(18):
                source.classifier.weight[row].fill_(row + 1)
                source.classifier.bias[row] = row + 101
        payload = {
            "state_dict": source.state_dict(),
            "source": {"model": medical.SOURCE_MODEL,
                       "pathologies": list(medical.SOURCE_PATHOLOGIES),
                       "valid_indices": list(medical.VALID_SOURCE_INDICES)},
        }
        self.assertEqual(medical.TARGET_SOURCE_ROWS, (17, 3, 1, 8, 4, 10, 0, 16, 7, None))
        with tempfile.TemporaryDirectory(prefix="cxr_medical_test_") as temporary:
            checkpoint = Path(temporary) / "safe_state_dict.pt"
            torch.save(payload, checkpoint)
            with patch.object(medical, "PRETRAINED_PATH", checkpoint):
                model = medical.build_chexpert_densenet(pretrained=True)
            for target, row in enumerate(medical.TARGET_SOURCE_ROWS[:-1]):
                self.assertTrue(torch.equal(model.classifier.weight[target], source.classifier.weight[row]))
                self.assertTrue(torch.equal(model.classifier.bias[target], source.classifier.bias[row]))
            self.assertLess(float(model.classifier.weight[-1].detach().abs().max()), 1.0)
            self.assertTrue(torch.equal(model.features.conv0[1].weight, source.features.conv0.weight))
            payload["source"]["pathologies"][0] = "Incorrect pathology"
            torch.save(payload, checkpoint)
            with patch.object(medical, "PRETRAINED_PATH", checkpoint):
                with self.assertRaisesRegex(ValueError, "pathologies"):
                    medical.build_chexpert_densenet(pretrained=True)

    def test_missing_safe_source_reports_actionable_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cxr_medical_missing_") as temporary:
            missing = Path(temporary) / "missing_state_dict.pt"
            with patch.object(medical, "PRETRAINED_PATH", missing):
                with self.assertRaisesRegex(FileNotFoundError, "pretrained=False"):
                    medical.build_chexpert_densenet(pretrained=True)


if __name__ == "__main__":
    unittest.main()
