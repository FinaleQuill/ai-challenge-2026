import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from cxr.data import CXRDataset, parse_targets, read_frame


class DataContractTests(unittest.TestCase):
    def test_multilabel_and_invalid_tokens(self):
        self.assertEqual(parse_targets("7;6;9"), [0, 0, 0, 0, 0, 1, 1, 0, 1, 0])
        for invalid in ("", "0", "11", "1;1", "1,2", "None", "-1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_targets(invalid)

    def test_deployment_needs_only_test_csv_and_keeps_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({"Dicom_id": ["00003"], "Subject_id": ["00001"],
                          "Study_id": ["00002"]}).to_csv(root / "test.csv", index=False)
            image_path = root / "test" / "00001" / "00002" / "00003.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("L", (320, 640), color=128).save(image_path)
            frame = read_frame(root, "test")
            self.assertEqual(frame.loc[0, "Subject_id"], "00001")
            self.assertNotIn("targets", frame)
            dataset = CXRDataset(frame, image_size=64, training=False)
            first, second = dataset[0], dataset[0]
            self.assertEqual(tuple(first["image"].shape), (3, 64, 64))
            self.assertTrue(torch.equal(first["image"], second["image"]))
            self.assertEqual(first["target"].sum().item(), 0)

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({"Dicom_id": ["picture"], "Subject_id": ["../outside"],
                          "Study_id": ["study"]}).to_csv(root / "test.csv", index=False)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                read_frame(root, "test")


if __name__ == "__main__":
    unittest.main()
