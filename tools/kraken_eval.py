"""kraken ile sentetik GT sayfalarında CER: ön-ikilileştirme × segmentasyon karşılaştırması.

Kullanım:
    venv/bin/python tools/kraken_eval.py --gt-dir evaluation/synthetic \\
        --kraken ~/projects/venvs/kraken/bin/kraken \\
        --model ~/projects/venvs/kraken_models/ottoman_best.mlmodel --out out_<tarih>_kraken

Hatlar (her sayfa için):
    sauvola+box   : preprocessing.binarize('sauvola') → kraken segment -x (kutu) → ocr
    nlbin+box     : kraken binarize (nlbin) → segment -x → ocr
    gray+baseline : ham gri → kraken segment -bl (blla) → ocr
Tümü `ocr --base-dir R` ile koşar (OpenITI modelleri BiDi'siz mantıksal sıra ister;
varsayılan reorder çıktıyı tersine çevirir, 2026-09-26'da ölçüldü).
Kraken ayrı venv'de (torch); bu betik yalnız alt süreç çağırır.
"""
import argparse
import os
import subprocess
import sys
import time
from glob import glob

import cv2 as cv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from evaluate import cer, normalize, find_pairs  # noqa: E402
from preprocessing import binarize  # noqa: E402

PERSIAN = str.maketrans({"ک": "ك", "ی": "ي", "ٓ": "", "ﻻ": "لا"})


def norm(t):
    return normalize(t.translate(PERSIAN), arabic=True)


def run_kraken(kraken, img, out_txt, seg_args, model, extra=()):
    cmd = [kraken, *extra, "-i", img, out_txt, *seg_args, "ocr", "-m", model, "--base-dir", "R"]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    ok = p.returncode == 0 and os.path.exists(out_txt)
    return ok, time.time() - t, (p.stderr or "")[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--kraken", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.path.join(_ROOT, f"out_{time.strftime('%Y%m%d')}_kraken"))
    ap.add_argument("--pipelines", nargs="+", default=["sauvola+box", "nlbin+box", "gray+baseline"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for img, gt in find_pairs(a.gt_dir):
        stem = os.path.splitext(os.path.basename(img))[0]
        ref = norm(open(gt, encoding="utf8").read())
        for pl in a.pipelines:
            src = img
            extra = ()
            if pl == "sauvola+box":
                src = os.path.join(a.out, f"{stem}_sauvola.png")
                cv.imwrite(src, 255 - binarize(cv.imread(img, 0), "sauvola"))
                seg = ["segment", "-x", "-d", "horizontal-rl"]
            elif pl == "nlbin+box":
                seg = ["binarize", "segment", "-x", "-d", "horizontal-rl"]
            elif pl == "gray+baseline":
                seg = ["segment", "-bl"]
            else:
                raise SystemExit(f"bilinmeyen hat: {pl}")
            out_txt = os.path.join(a.out, f"{stem}__{pl}.txt")
            ok, dt, err = run_kraken(a.kraken, src, out_txt, seg, a.model, extra)
            lines = [l for l in open(out_txt, encoding="utf8").read().split("\n") if l.strip()] if ok else []
            hyp = norm("".join(lines))
            e, n, r = cer(ref, hyp)
            rows.append((stem, pl, ok, len(lines), n, e, r, dt))
            print(f"{stem:24s} {pl:14s} ok={ok!s:5s} satır={len(lines):2d} CER={r:.4f} {dt:.1f}s {err.strip().splitlines()[-1] if (not ok and err.strip()) else ''}", flush=True)
    md = ["# kraken CER raporu", "", f"- GT: `{a.gt_dir}` · model: `{os.path.basename(a.model)}` · `ocr --base-dir R`", "",
          "## Toplu (micro)", "", "| hat | sayfa | başarısız | ref. kar. | düzenleme | **CER** | süre (s) |", "|---|---|---|---|---|---|---|"]
    for pl in a.pipelines:
        rs = [r for r in rows if r[1] == pl]
        n = sum(r[4] for r in rs); e = sum(r[5] for r in rs)
        md.append(f"| {pl} | {len(rs)} | {sum(1 for r in rs if not r[2])} | {n} | {e} | **{e / n:.4f}** | {sum(r[7] for r in rs):.0f} |")
    md += ["", "## Sayfa bazında", "", "| sayfa | hat | satır | CER | süre (s) |", "|---|---|---|---|---|"]
    md += [f"| {s} | {pl} | {nl} | {r:.4f} | {dt:.1f} |" for s, pl, ok, nl, n, e, r, dt in rows]
    open(os.path.join(a.out, "kraken_report.md"), "w", encoding="utf8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
