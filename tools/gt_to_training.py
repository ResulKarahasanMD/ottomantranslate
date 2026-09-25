"""Doldurulmuş lines.tsv + kraken seg.json → PageXML eğitim verisi → ketos ince ayarı.

Kullanım:
    venv/bin/python tools/gt_to_training.py --lines-dir out_20260926_nfs_lines \\
        --out out_20260926_nfs_train [--val-pages NFS_d___12369_00008] [--val-ratio 0.15] \\
        [--include-red] [--train --ketos ~/projects/venvs/kraken/bin/ketos \\
         --base-model ~/projects/venvs/kraken_models/ottoman_best.mlmodel --epochs 30]

Adımlar:
  1. lines.tsv'den gt'si dolu satırları al. Yalnız "?" olan satır atlanır (okunamadı);
     "[kırmızı]" ile başlayanlar --include-red verilmedikçe atlanır (kırmızı mürekkep,
     ayrı dağılım). Satır içindeki "?" işaretleri korunmaz: "?" içeren satır da atlanır,
     çünkü belirsiz harf eğitimi bozar. Atlanan sayılar raporlanır.
  2. Her sayfa için PageXML (PcGts 2019) yazılır: TextLine Coords = seg.json boundary
     poligonu, Baseline = seg.json baseline, TextEquiv/Unicode = gt (NFC).
  3. train.txt / val.txt: --val-pages verilirse sayfa bazlı, yoksa --val-ratio ile
     satır bazlı değil SAYFA bazlı rastgele bölme (aynı sayfanın satırları sızmasın).
  4. --train: `ketos train -f page -i <base> --resize union` (yeni kod noktaları eklenir,
     mevcutlar korunur) ardından `ketos test` ile val CER'i; komutlar ve çıktılar
     <out>/train.log'a yazılır. ketos ayrı venv'de, alt süreçle çağrılır.
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from xml.sax.saxutils import escape

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def load_rows(lines_dir, include_red):
    rows = list(csv.DictReader(open(os.path.join(lines_dir, "lines.tsv"), encoding="utf8"), delimiter="\t"))
    pages = {r["page"]: r for r in csv.DictReader(open(os.path.join(lines_dir, "pages.tsv"), encoding="utf8"), delimiter="\t")}
    kept, skipped = defaultdict(list), defaultdict(int)
    for r in rows:
        gt = unicodedata.normalize("NFC", (r.get("gt") or "").strip())
        if not gt:
            skipped["boş"] += 1
            continue
        if gt.startswith("[kırmızı]"):
            if not include_red:
                skipped["kırmızı"] += 1
                continue
            gt = gt[len("[kırmızı]"):].strip()
        if "?" in gt or not gt:
            skipped["belirsiz(?)"] += 1
            continue
        kept[r["page"]].append((int(r["idx"]), gt))
    return kept, dict(skipped), pages


def pts(poly):
    return " ".join(f"{int(x)},{int(y)}" for x, y in poly)


def write_pagexml(page, img_path, w, h, seg, lines, out_xml):
    by_idx = {i: g for i, g in lines}
    tl = []
    for idx, line in enumerate(seg["lines"]):
        if idx not in by_idx:
            continue
        if "baseline" not in line:  # kutu segmentasyonu (bkz. nfs_lines_export.py)
            x0, y0, x1, y1 = line["bbox"]
            line["boundary"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            yb = y1 - max(2, (y1 - y0) // 5)
            line["baseline"] = [[x0, yb], [x1, yb]]
        if not line.get("boundary"):
            continue
        tl.append(
            f'      <TextLine id="l{idx}">\n'
            f'        <Coords points="{pts(line["boundary"])}"/>\n'
            f'        <Baseline points="{pts(line["baseline"])}"/>\n'
            f'        <TextEquiv><Unicode>{escape(by_idx[idx])}</Unicode></TextEquiv>\n'
            f'      </TextLine>')
    xml = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<PcGts xmlns="{PAGE_NS}">\n'
           f'  <Metadata><Creator>gt_to_training.py</Creator><Created>{time.strftime("%Y-%m-%dT%H:%M:%S")}</Created></Metadata>\n'
           f'  <Page imageFilename="{escape(img_path)}" imageWidth="{w}" imageHeight="{h}">\n'
           f'    <TextRegion id="r0"><Coords points="0,0 {w},0 {w},{h} 0,{h}"/>\n' + "\n".join(tl) +
           f'\n    </TextRegion>\n  </Page>\n</PcGts>\n')
    open(out_xml, "w", encoding="utf8").write(xml)
    return len(tl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-pages", nargs="*", default=None)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--include-red", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--ketos", default=os.path.expanduser("~/projects/venvs/kraken/bin/ketos"))
    ap.add_argument("--base-model", default=os.path.expanduser("~/projects/venvs/kraken_models/ottoman_best.mlmodel"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lrate", type=float, default=0.0001)
    a = ap.parse_args()

    kept, skipped, pages = load_rows(a.lines_dir, a.include_red)
    xml_dir = os.path.join(a.out, "pagexml")
    os.makedirs(xml_dir, exist_ok=True)
    counts = {}
    for page, lines in kept.items():
        seg = json.load(open(os.path.join(a.lines_dir, page, "seg.json"), encoding="utf8"))
        pg = pages[page]
        counts[page] = write_pagexml(page, pg["image"], pg["width"], pg["height"], seg, lines,
                                     os.path.join(xml_dir, f"{page}.xml"))
    total = sum(counts.values())
    print(f"GT'li satır: {total}  sayfa: {len(counts)}  atlanan: {skipped}")
    if not total:
        raise SystemExit("Eğitilecek satır yok: lines.tsv'de gt sütunu boş. Önce sheet.html'de doldur.")

    page_list = sorted(counts)
    if a.val_pages:
        val = [p for p in page_list if p in set(a.val_pages)]
    else:
        rnd = random.Random(a.seed)
        rnd.shuffle(page_list)
        n_val = max(1, round(len(page_list) * a.val_ratio)) if len(page_list) > 1 else 0
        val = sorted(page_list[:n_val])
    train = sorted(p for p in counts if p not in set(val))
    for name, plist in (("train", train), ("val", val)):
        with open(os.path.join(a.out, f"{name}.txt"), "w") as f:
            f.write("".join(os.path.abspath(os.path.join(xml_dir, f"{p}.xml")) + "\n" for p in plist))
        print(f"{name}: {len(plist)} sayfa, {sum(counts[p] for p in plist)} satır")

    model_dir = os.path.join(a.out, "model")
    # ketos 7: cihaz/iş parçacığı üst komut seçeneği; -t/-e liste dosyaları.
    # DİKKAT: ketos train/test'te BiDi yeniden sıralama VARSAYILAN kalmalı (GT mantıksal
    # sırada yazılır, ketos görüntü sırasına çevirir). `--no-reorder --base-dir R` ile
    # taban modelin val doğruluğu %95'ten %15'e düşüyor (2026-09-26 ölçümü). `--base-dir R`
    # yalnız `kraken ocr` tarafında gerekir.
    cmd_train = [a.ketos, "-d", a.device, "--threads", "4", "train", "-f", "page",
                 "-i", a.base_model, "--resize", "union",
                 "-o", os.path.join(model_dir, "ft"), "-N", str(a.epochs), "-q", "fixed",
                 "-r", str(a.lrate), "-B", "4",
                 "-e", os.path.join(a.out, "val.txt"), "-t", os.path.join(a.out, "train.txt")]
    cmd_test = [a.ketos, "-d", a.device, "test", "-f", "page", "-m", "<model>",
                "-e", os.path.join(a.out, "val.txt")]
    print("ketos train:", " ".join(cmd_train))
    print("ketos test :", " ".join(cmd_test))
    if not a.train:
        return
    os.makedirs(model_dir, exist_ok=True)
    log = open(os.path.join(a.out, "train.log"), "a", encoding="utf8")
    log.write("\n== " + " ".join(cmd_train) + "\n")
    t = time.time()
    p = subprocess.run(cmd_train, capture_output=True, text=True)
    log.write(p.stdout + p.stderr)
    print(f"ketos train bitti: rc={p.returncode} {time.time() - t:.0f}s; son satırlar:")
    print("\n".join((p.stdout + p.stderr).strip().splitlines()[-6:]))
    ft_dir = os.path.join(model_dir, "ft")
    cands = sorted(glob.glob(os.path.join(ft_dir, "best_*.safetensors")) + glob.glob(os.path.join(ft_dir, "*best*.mlmodel")))
    if not cands:
        raise SystemExit(f"en iyi model bulunamadı ({ft_dir}); bkz. {a.out}/train.log")
    best = cands[-1]
    print("en iyi model:", best)
    for label, m in (("taban", a.base_model), ("ince-ayar", best)):
        cmd = [c if c != "<model>" else m for c in cmd_test]
        log.write("\n== " + " ".join(cmd) + "\n")
        q = subprocess.run(cmd, capture_output=True, text=True)
        log.write(q.stdout + q.stderr)
        acc = [l for l in (q.stdout + q.stderr).splitlines() if "ccuracy" in l or "rror" in l]
        print(f"val [{label}]:", " | ".join(acc[-3:]) if acc else (q.stdout + q.stderr).strip().splitlines()[-1:])


if __name__ == "__main__":
    main()
