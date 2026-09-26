"""Temiz sentetik sayfalara arşiv benzeri bozulmalar uygular (deterministik).

Kullanım:
    venv/bin/python tools/degrade.py <temiz_dizin> <çıktı_dizin>

Her <ad>.png için şu türevler üretilir (gt.txt kopyalanır):
    <ad>_clean       değiştirilmemiş
    <ad>_gradient    yatay+dikey aydınlatma düşüşü (gölgeli tarama)
    <ad>_stain       düşük frekanslı koyu lekeler (sararma / foxing)
    <ad>_lowcontrast soluk mürekkep + Gauss gürültüsü + hafif bulanıklık
    <ad>_combo       gradient + stain + gürültü
Global Otsu'nun zorlanacağı, yerel eşiğin (Sauvola) hedeflediği koşullar.
"""
import os
import shutil
import sys
from glob import glob

import cv2 as cv
import numpy as np


def _gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img


def gradient(gray, rng):
    h, w = gray.shape
    x = np.linspace(1.0, 0.45, w)[None, :]
    y = np.linspace(1.0, 0.8, h)[:, None]
    illum = x * y
    return np.clip(gray.astype(np.float32) * illum, 0, 255).astype(np.uint8)


def stain(gray, rng, n=6):
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    field = np.ones((h, w), np.float32)
    for _ in range(n):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sx, sy = rng.uniform(w * 0.08, w * 0.25), rng.uniform(h * 0.15, h * 0.5)
        depth = rng.uniform(0.35, 0.6)  # 1-depth kadar karartma
        blob = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        field *= (1.0 - depth * blob)
    return np.clip(gray.astype(np.float32) * field, 0, 255).astype(np.uint8)


def lowcontrast(gray, rng):
    g = gray.astype(np.float32)
    # 0 -> 115 (soluk mürekkep), 255 -> 195 (grileşmiş kağıt)
    g = 115 + (g / 255.0) * 80
    g = cv.GaussianBlur(g, (3, 3), 0)
    g += rng.normal(0, 10, g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def combo(gray, rng):
    g = gradient(gray, rng)
    g = stain(g, rng, n=4)
    g = g.astype(np.float32) + rng.normal(0, 6, g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


VARIANTS = {
    "clean": lambda g, r: g,
    "gradient": gradient,
    "stain": stain,
    "lowcontrast": lowcontrast,
    "combo": combo,
}


def main(src, dst, seed=20260926):
    os.makedirs(dst, exist_ok=True)
    pngs = sorted(glob(os.path.join(src, "*.png")))
    if not pngs:
        raise SystemExit(f"{src} altında png yok")
    for p in pngs:
        stem = os.path.splitext(os.path.basename(p))[0]
        gt = os.path.join(src, stem + ".gt.txt")
        gray = _gray(cv.imread(p))
        for vi, (name, fn) in enumerate(VARIANTS.items()):
            # hash(str) süreçler arası rastgele (PYTHONHASHSEED); varyant sırası kararlı
            rng = np.random.default_rng(seed + vi)
            out = fn(gray, rng)
            out_bgr = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
            cv.imwrite(os.path.join(dst, f"{stem}_{name}.png"), out_bgr)
            shutil.copy(gt, os.path.join(dst, f"{stem}_{name}.gt.txt"))
    print(f"{len(pngs)} sayfa × {len(VARIANTS)} varyant → {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
