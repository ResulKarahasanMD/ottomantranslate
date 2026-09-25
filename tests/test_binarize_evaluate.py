"""Hızlı birim testleri (bağımlılık: yalnız unittest).

    venv/bin/python -m unittest tests.test_binarize_evaluate -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import binarize, binary_sauvola, BINARIZATION_METHODS  # noqa: E402
from evaluate import levenshtein, cer, wer, normalize  # noqa: E402


def _synthetic_page(gradient=False):
    """Beyaz zemin, iki koyu 'satır' bloğu; istenirse yatay aydınlatma düşüşü."""
    h, w = 200, 400
    img = np.full((h, w), 230, np.uint8)
    img[40:60, 30:370] = 20
    img[120:140, 30:370] = 20
    if gradient:
        ramp = np.linspace(1.0, 0.35, w)[None, :]
        img = np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)
    return img


class BinarizeTests(unittest.TestCase):
    def test_methods_listed(self):
        self.assertIn("otsu", BINARIZATION_METHODS)
        self.assertIn("sauvola", BINARIZATION_METHODS)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            binarize(_synthetic_page(), "niblack")

    def test_polarity_text_white(self):
        img = _synthetic_page()
        for m in BINARIZATION_METHODS:
            b = binarize(img, m)
            self.assertEqual(b.dtype, np.uint8)
            self.assertEqual(set(np.unique(b).tolist()) - {0, 255}, set())
            # Metin bölgesi beyaz, zemin siyah
            self.assertGreater((b[45:55, 100:300] == 255).mean(), 0.95, m)
            self.assertLess((b[80:110, 100:300] == 255).mean(), 0.05, m)

    def test_sauvola_survives_gradient_otsu_does_not(self):
        img = _synthetic_page(gradient=True)
        otsu = binarize(img, "otsu")
        sauv = binarize(img, "sauvola")
        # Karanlık (sağ) tarafta zemin: Otsu onu metin sayar, Sauvola saymaz
        self.assertGreater((otsu[80:110, 300:390] == 255).mean(), 0.5)
        self.assertLess((sauv[80:110, 300:390] == 255).mean(), 0.05)
        # Sauvola sağ taraftaki gerçek metni de korur
        self.assertGreater((sauv[45:55, 300:360] == 255).mean(), 0.9)

    def test_sauvola_even_window_and_speckle(self):
        img = _synthetic_page()
        rng = np.random.default_rng(0)
        noisy = img.copy()
        ys, xs = rng.integers(70, 110, 30), rng.integers(0, 400, 30)
        noisy[ys, xs] = 0  # tek piksel lekeler
        b = binary_sauvola(noisy, window_size=30, k=0.3)  # çift pencere düzeltilmeli
        self.assertEqual((b[70:110] == 255).sum(), 0)  # speckle atıldı
        b_keep = binary_sauvola(noisy, window_size=31, k=0.3, min_area=0)
        self.assertGreater((b_keep[70:110] == 255).sum(), 0)


class RedInkTests(unittest.TestCase):
    def test_sepia_black_kept_red_removed(self):
        from archive_preprocessing import remove_red_ink
        import cv2 as cv
        # NFS s.2 ölçümleri: kırmızı hue≈3,S≈110; sepya siyah hue≈17,S≈55; kağıt hue≈16,S≈27
        # Maske 3x3 genişletildiği için pikseller birbirinden uzak tutulur.
        hsv = np.zeros((7, 7, 3), np.uint8)
        hsv[..., :] = (16, 27, 220)  # kağıt
        hsv[0, 0] = (3, 110, 150)    # kırmızı mürekkep
        hsv[6, 6] = (17, 55, 60)     # sepya siyah mürekkep
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        out = remove_red_ink(bgr)
        self.assertEqual(tuple(out[0, 0]), (255, 255, 255))      # kırmızı beyazlandı
        self.assertTrue((out[6, 6] == bgr[6, 6]).all())          # siyah korundu
        self.assertTrue((out[3, 3] == bgr[3, 3]).all())          # kağıt korundu


class MetricTests(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(levenshtein("", ""), 0)
        self.assertEqual(levenshtein("abc", ""), 3)
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein(["a", "b"], ["a", "c", "b"]), 1)

    def test_cer_wer(self):
        self.assertEqual(cer("كتاب", "كتاب"), (0, 4, 0.0))
        self.assertEqual(cer("كتاب", "كتب")[2], 0.25)
        self.assertEqual(wer("بو كتاب", "بو كتب")[2], 0.5)
        self.assertIsNone(cer("", "x")[2])

    def test_normalize(self):
        self.assertEqual(normalize("بو  كتاب\n"), "بوكتاب")
        self.assertEqual(normalize("بو كتاب", keep_spaces=True), "بو كتاب")
        self.assertEqual(normalize("أإآ ة ى", arabic=True), "اااهي")
        self.assertEqual(normalize("كِتَابْ"), "كتاب")  # hareke atılır
        self.assertEqual(normalize("كـتاب"), "كتاب")   # tatvil atılır


if __name__ == "__main__":
    unittest.main()
