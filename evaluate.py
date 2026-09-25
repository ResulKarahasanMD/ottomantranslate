"""OCR değerlendirme: CER / WER hesabı ve ikilileştirme yöntemlerini karşılaştırma.

Kullanım:
    venv/bin/python evaluate.py --gt-dir evaluation/synthetic \\
        --methods otsu sauvola --out out_20260926_cer

Girdi dizininde her görüntü için aynı adlı `.gt.txt` bulunmalıdır:
    sayfa01.png  +  sayfa01.gt.txt

CER = Levenshtein(ref, hyp) / len(ref)   (karakter düzeyi, boşluklar dahil
      değildir: --keep-spaces verilmedikçe iki taraftan da boşluk atılır)
WER = Levenshtein(ref_kelimeler, hyp_kelimeler) / len(ref_kelimeler)

Toplu (micro) CER = toplam düzenleme / toplam referans karakteri; sayfalar
uzunluklarıyla ağırlıklanır.

NOT: ocr.run() multiprocessing.Pool kullanır; bu betik dosyadan çalıştırılmalı,
stdin/heredoc ile değil (macOS spawn tuzağı).
"""

import argparse
import json
import os
import sys
import time
import unicodedata
from glob import glob

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MODULE_DIR)

IMAGE_EXTS = ("png", "jpg", "jpeg", "bmp")

# Arapça normalizasyon: elif türevleri -> ا, te merbuta -> ه, elif maksure -> ي.
# Model 28 temel harfle eğitildiği için bu birleştirme model sınıflarıyla uyumludur.
_ARABIC_NORM = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي",
})
# Hareke ve tatvil işaretleri (Mn sınıfı + ـ)
_STRIP_MARKS = True


def normalize(text: str, arabic: bool = False, keep_spaces: bool = False) -> str:
    """Karşılaştırma öncesi metni sadeleştirir."""
    text = unicodedata.normalize("NFC", text)
    if _STRIP_MARKS:
        text = "".join(ch for ch in text
                       if unicodedata.category(ch) != "Mn" and ch != "ـ")
    if arabic:
        text = text.translate(_ARABIC_NORM)
    if keep_spaces:
        return " ".join(text.split())
    return "".join(text.split())


def levenshtein(a, b) -> int:
    """İki dizi (str veya list) arasındaki düzenleme uzaklığı, O(len(a)*len(b))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1,        # silme
                           cur[j - 1] + 1,     # ekleme
                           prev[j - 1] + cost))  # değiştirme
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> tuple:
    """(düzenleme, referans uzunluğu, oran). Boş referansta oran None."""
    d = levenshtein(ref, hyp)
    n = len(ref)
    return d, n, (d / n if n else None)


def wer(ref: str, hyp: str) -> tuple:
    r, h = ref.split(), hyp.split()
    d = levenshtein(r, h)
    n = len(r)
    return d, n, (d / n if n else None)


def find_pairs(gt_dir):
    pairs = []
    for ext in IMAGE_EXTS:
        for img in sorted(glob(os.path.join(gt_dir, f"*.{ext}"))):
            stem = os.path.splitext(img)[0]
            gt = stem + ".gt.txt"
            if os.path.exists(gt):
                pairs.append((img, gt))
    return pairs


def evaluate(gt_dir, methods, arabic_norm, keep_spaces, archive_mode, bin_kwargs):
    from ocr import run

    pairs = find_pairs(gt_dir)
    if not pairs:
        raise SystemExit(f"'{gt_dir}' altında görüntü + .gt.txt çifti yok.")

    rows = []
    for img, gt in pairs:
        with open(gt, encoding="utf8") as f:
            ref_raw = f.read()
        ref_c = normalize(ref_raw, arabic_norm, keep_spaces=False)
        ref_w = normalize(ref_raw, arabic_norm, keep_spaces=True)
        for m in methods:
            t0 = time.time()
            name, _, n_words, hyp_raw = run(img, archive_mode=archive_mode,
                                            binarization=m, write_output=False,
                                            **bin_kwargs.get(m, {}))
            dt = time.time() - t0
            hyp_raw = hyp_raw or ""
            hyp_c = normalize(hyp_raw, arabic_norm, keep_spaces=False)
            hyp_w = normalize(hyp_raw, arabic_norm, keep_spaces=True)
            ce, cn, cr = cer(ref_c, hyp_c)
            we, wn, wr = wer(ref_w, hyp_w)
            rows.append({
                "image": os.path.basename(img), "method": m,
                "ok": name is not None,
                "ref_chars": cn, "hyp_chars": len(hyp_c), "char_edits": ce, "cer": cr,
                "ref_words": wn, "hyp_words": len(hyp_w.split()), "word_edits": we, "wer": wr,
                "segments": n_words, "seconds": round(dt, 2),
                "hyp": hyp_raw,
            })
            print(f"{os.path.basename(img):40s} {m:8s} CER={cr if cr is None else round(cr, 4)} "
                  f"WER={wr if wr is None else round(wr, 4)} seg={n_words} {dt:.1f}s", flush=True)
    return rows


def summarize(rows, methods):
    summary = []
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        ce = sum(r["char_edits"] for r in rs)
        cn = sum(r["ref_chars"] for r in rs)
        we = sum(r["word_edits"] for r in rs)
        wn = sum(r["ref_words"] for r in rs)
        summary.append({
            "method": m, "images": len(rs), "failed": sum(1 for r in rs if not r["ok"]),
            "ref_chars": cn, "char_edits": ce, "micro_cer": (ce / cn if cn else None),
            "ref_words": wn, "word_edits": we, "micro_wer": (we / wn if wn else None),
            "seconds_total": round(sum(r["seconds"] for r in rs), 1),
        })
    return summary


def _fmt(x):
    return "—" if x is None else f"{x:.4f}"


def to_markdown(rows, summary, args):
    lines = ["# CER / WER raporu", "",
             f"- Girdi: `{args.gt_dir}`  ·  yöntemler: {', '.join(args.methods)}",
             f"- Arapça normalizasyon: {args.arabic_norm}  ·  boşluk sayımı: {args.keep_spaces}"
             f"  ·  arşiv modu: {args.archive_mode}",
             f"- Sauvola parametreleri: {json.dumps(args.bin_kwargs.get('sauvola', {}))}",
             "", "## Toplu (micro)", "",
             "| yöntem | görüntü | başarısız | ref. karakter | düzenleme | **CER** | ref. kelime | düzenleme | **WER** | süre (s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['method']} | {s['images']} | {s['failed']} | {s['ref_chars']} | "
                     f"{s['char_edits']} | **{_fmt(s['micro_cer'])}** | {s['ref_words']} | "
                     f"{s['word_edits']} | **{_fmt(s['micro_wer'])}** | {s['seconds_total']} |")
    lines += ["", "## Görüntü bazında", "",
              "| görüntü | yöntem | ref. kar. | çıktı kar. | CER | WER | segment | süre (s) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['image']} | {r['method']} | {r['ref_chars']} | {r['hyp_chars']} | "
                     f"{_fmt(r['cer'])} | {_fmt(r['wer'])} | {r['segments']} | {r['seconds']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True, help="görüntü + .gt.txt çiftlerinin dizini")
    ap.add_argument("--methods", nargs="+", default=["otsu", "sauvola"])
    ap.add_argument("--out", default=None, help="rapor dizini (varsayılan out_<tarih>_cer)")
    ap.add_argument("--arabic-norm", action="store_true", help="elif/te merbuta/ye birleştir")
    ap.add_argument("--keep-spaces", action="store_true", help="CER'de boşlukları da say")
    ap.add_argument("--archive-mode", action="store_true")
    ap.add_argument("--sauvola-window", type=int, default=31)
    ap.add_argument("--sauvola-k", type=float, default=0.3)
    args = ap.parse_args(argv)
    args.bin_kwargs = {"sauvola": {"window_size": args.sauvola_window, "k": args.sauvola_k}}

    out = args.out or os.path.join(_MODULE_DIR, f"out_{time.strftime('%Y%m%d')}_cer")
    os.makedirs(out, exist_ok=True)

    rows = evaluate(args.gt_dir, args.methods, args.arabic_norm, args.keep_spaces,
                    args.archive_mode, args.bin_kwargs)
    summary = summarize(rows, args.methods)

    with open(os.path.join(out, "cer_rows.json"), "w", encoding="utf8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    md = to_markdown(rows, summary, args)
    with open(os.path.join(out, "cer_report.md"), "w", encoding="utf8") as f:
        f.write(md)
    print()
    print(md)
    print(f"Rapor: {out}/cer_report.md")


if __name__ == "__main__":
    main()
