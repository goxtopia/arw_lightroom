import unittest
import numpy as np
import os
import sys

# Add the project root to the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import ImagePipeline
from app.processors import (
    ExposureProcessor, ContrastProcessor, SaturationProcessor,
    HighlightsShadowsProcessor, CurveProcessor
)

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create a mock 10x10 RGB image with random values in range [0, 1]
        self.mock_img = np.random.rand(10, 10, 3).astype(np.float32)

    def test_exposure(self):
        proc = ExposureProcessor()
        # Test 0 EV (should not change image)
        out = proc.process(self.mock_img, {"exposure": 0.0})
        np.testing.assert_array_almost_equal(self.mock_img, out)
        
        # Test +1 EV (should double the values, capped at 1.0)
        out_ev1 = proc.process(self.mock_img, {"exposure": 1.0})
        expected = np.clip(self.mock_img * 2.0, 0.0, 1.0)
        np.testing.assert_array_almost_equal(expected, out_ev1)

    def test_contrast(self):
        proc = ContrastProcessor()
        # Test 0 contrast (no change)
        out = proc.process(self.mock_img, {"contrast": 0.0})
        np.testing.assert_array_almost_equal(self.mock_img, out)
        
        # Test positive contrast
        out_pos = proc.process(self.mock_img, {"contrast": 0.5})
        self.assertTrue(np.all(out_pos >= 0.0))
        self.assertTrue(np.all(out_pos <= 1.0))
        # Values > 0.5 should increase, values < 0.5 should decrease
        mask_greater = self.mock_img > 0.5
        mask_less = self.mock_img < 0.5
        self.assertTrue(np.all(out_pos[mask_greater] >= self.mock_img[mask_greater]))
        self.assertTrue(np.all(out_pos[mask_less] <= self.mock_img[mask_less]))

    def test_saturation(self):
        proc = SaturationProcessor()
        # Test 0 saturation (no change)
        out = proc.process(self.mock_img, {"saturation": 0.0})
        np.testing.assert_array_almost_equal(self.mock_img, out)
        
        # Test -1.0 saturation (should produce a grayscale image where channels are identical)
        out_mono = proc.process(self.mock_img, {"saturation": -1.0})
        self.assertTrue(np.allclose(out_mono[:, :, 0], out_mono[:, :, 1]))
        self.assertTrue(np.allclose(out_mono[:, :, 0], out_mono[:, :, 2]))

    def test_highlights_shadows(self):
        proc = HighlightsShadowsProcessor()
        # Test 0 (no change)
        out = proc.process(self.mock_img, {"highlights": 0.0, "shadows": 0.0})
        np.testing.assert_array_almost_equal(self.mock_img, out)
        
        # Test shadow recovery (should lift shadows)
        out_shadows = proc.process(self.mock_img, {"highlights": 0.0, "shadows": 1.0})
        self.assertTrue(np.all(out_shadows >= self.mock_img))

    def test_curve(self):
        proc = CurveProcessor()
        # Test no curve (no change)
        out = proc.process(self.mock_img, {})
        np.testing.assert_array_almost_equal(self.mock_img, out)
        
        # Test linear curve (no change)
        linear_params = {
            "curves": {
                "rgb": [[0.0, 0.0], [1.0, 1.0]]
            }
        }
        out_linear = proc.process(self.mock_img, linear_params)
        np.testing.assert_array_almost_equal(self.mock_img, out_linear, decimal=3)

    def test_pipeline_chain(self):
        # Chain several processors together
        pipe = ImagePipeline([
            ExposureProcessor(),
            ContrastProcessor(),
            SaturationProcessor()
        ])
        
        params = {
            "exposure": 0.5,
            "contrast": 0.2,
            "saturation": -0.5
        }
        
        out = pipe.run(self.mock_img, params)
        self.assertEqual(out.shape, self.mock_img.shape)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))

if __name__ == "__main__":
    unittest.main()
