"""Arşiv sayfalarını kraken ile satırlara böl, satır kırpmalarını ve transkripsiyon
sayfasını üret (kraken ince ayarı için ground truth hazırlığı).

Kullanım:
    venv/bin/python tools/nfs_lines_export.py --out out_20260926_nfs_lines \\
        --kraken ~/projects/venvs/kraken/bin/kraken \\
        --seg-model ~/projects/venvs/kraken_models/seg_general.mlmodel \\
        "~/Downloads/NFS.d.._12369-_25.07.2026 06:05:11"/NFS_d___12369_0000*.jpg

Çıktı (sayfa başına <out>/<sayfa>/):
    seg.json          kraken segmentasyonu (taban çizgisi + poligon), ham
    lines/NNN.png     satır kırpması (poligon dışı beyaz, kenar payı --pad)
    overlay.jpg       taban çizgileri ve satır numaraları (kontrol için)
Toplu:
    lines.tsv         page, idx, x0,y0,x1,y1, baseline_px, gt (BOŞ — elle doldurulur)
    sheet.html        her kırpma için RTL metin alanı; tarayıcıda doldurulur,
                      "TSV indir" ile gt sütunu dolu lines.tsv üretir (taslak
                      localStorage'da tutulur).
Kural: okunamayan satıra "?" yaz, tahmin etme; kırmızı sayı/çapraz not satırı ise
"[kırmızı] …" ile başlat. Boş bırakılan satır GT'ye girmez.
"""
import argparse
import html
import json
import os
import subprocess
import sys
import time

import cv2 as cv
import numpy as np


def segment(kraken, seg_model, img_path, out_json):
    cmd = [kraken, "-i", img_path, out_json, "segment", "-bl", "-i", seg_model]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0 or not os.path.exists(out_json):
        raise RuntimeError(f"kraken segment başarısız: {img_path}\n{p.stderr[-500:]}")
    return json.load(open(out_json, encoding="utf8"))


def crop_line(img, boundary, pad):
    h, w = img.shape[:2]
    poly = np.array(boundary, np.int32)
    x0, y0 = np.maximum(poly.min(axis=0) - pad, 0)
    x1, y1 = np.minimum(poly.max(axis=0) + pad, [w - 1, h - 1])
    region = img[y0:y1 + 1, x0:x1 + 1].copy()
    mask = np.zeros(region.shape[:2], np.uint8)
    cv.fillPoly(mask, [poly - [x0, y0]], 255)
    mask = cv.dilate(mask, np.ones((pad + 1, pad + 1), np.uint8))
    region[mask == 0] = 255
    return region, (int(x0), int(y0), int(x1), int(y1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kraken", required=True)
    ap.add_argument("--seg-model", required=True)
    ap.add_argument("--pad", type=int, default=12)
    ap.add_argument("--min-baseline", type=int, default=0, help="bundan kısa taban çizgileri atlanır (px); 0 = hepsi")
    ap.add_argument("--short-px", type=int, default=60, help="bundan kısa taban çizgileri sayfada 'kısa' etiketi alır (kırmızı sayı/parça)")
    ap.add_argument("--force", action="store_true", help="mevcut seg.json'u yok say, yeniden segmente et")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for img_path in a.images:
        page = os.path.splitext(os.path.basename(img_path))[0]
        pdir = os.path.join(a.out, page)
        os.makedirs(os.path.join(pdir, "lines"), exist_ok=True)
        t = time.time()
        seg_json = os.path.join(pdir, "seg.json")
        if os.path.exists(seg_json) and not a.force:
            seg = json.load(open(seg_json, encoding="utf8"))  # önbellek: yeniden segmentasyon yok
        else:
            seg = segment(a.kraken, a.seg_model, img_path, seg_json)
        img = cv.imread(img_path)
        overlay = img.copy()
        kept = 0
        for idx, line in enumerate(seg.get("lines", [])):
            bl = np.array(line["baseline"], np.int32)
            length = int(np.sum(np.linalg.norm(np.diff(bl, axis=0), axis=1)))
            if length < a.min_baseline or not line.get("boundary"):
                continue
            crop, box = crop_line(img, line["boundary"], a.pad)
            fn = os.path.join("lines", f"{idx:03d}.png")
            cv.imwrite(os.path.join(pdir, fn), crop)
            cv.polylines(overlay, [bl.reshape(-1, 1, 2)], False, (0, 200, 0), 4)
            cv.putText(overlay, str(idx), tuple(bl[0]), cv.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 3)
            rows.append({"page": page, "idx": idx, "box": box, "baseline_px": length,
                         "img": f"{page}/{fn}"})
            kept += 1
        cv.imwrite(os.path.join(pdir, "overlay.jpg"), overlay, [cv.IMWRITE_JPEG_QUALITY, 80])
        print(f"{page}: {len(seg.get('lines', []))} satır bulundu, {kept} kırpıldı, {time.time() - t:.0f}s", flush=True)

    with open(os.path.join(a.out, "lines.tsv"), "w", encoding="utf8") as f:
        f.write("page\tidx\tx0\ty0\tx1\ty1\tbaseline_px\tshort\timg\tgt\n")
        for r in rows:
            f.write(f"{r['page']}\t{r['idx']}\t{r['box'][0]}\t{r['box'][1]}\t{r['box'][2]}\t{r['box'][3]}\t"
                    f"{r['baseline_px']}\t{int(r['baseline_px'] < a.short_px)}\t{r['img']}\t\n")

    items = "\n".join(
        f'<div class="line{" short" if r["baseline_px"] < a.short_px else ""}" data-key="{r["page"]}/{r["idx"]}">'
        f'<div class="meta">{html.escape(r["page"])} · #{r["idx"]} · {r["baseline_px"]} px'
        f'{" · <b>kısa</b> (kırmızı sayı / parça olabilir)" if r["baseline_px"] < a.short_px else ""}</div>'
        f'<img src="{html.escape(r["img"])}" loading="lazy">'
        f'<textarea dir="rtl" lang="ota" rows="1" placeholder="okunamıyorsa ? yaz"></textarea></div>'
        for r in rows)
    sheet = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>NFS satır transkripsiyonu</title>
<style>
body{{font-family:system-ui;margin:16px;background:#f6f4ee;color:#222}}
.line{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;margin:8px 0}}
.meta{{font-size:12px;color:#777;margin-bottom:4px}}
img{{max-width:100%;display:block;border:1px solid #eee}}
textarea{{width:100%;box-sizing:border-box;margin-top:6px;font-size:26px;font-family:"Geeza Pro","Al Nile",serif;
  padding:6px;border:1px solid #bbb;border-radius:4px}}
.bar{{position:sticky;top:0;background:#f6f4ee;padding:8px 0;border-bottom:1px solid #ccc;z-index:1}}
.short .meta{{color:#a55}} body.hideshort .short{{display:none}}
button{{font-size:14px;padding:6px 12px}} #stat{{margin-left:12px;color:#555}}
</style></head><body>
<div class="bar"><button id="dl">TSV indir (gt dolu)</button><button id="clr">Taslağı temizle</button>
<label style="margin-left:12px"><input type="checkbox" id="hs"> kısa satırları gizle (&lt;{a.short_px} px)</label>
<span id="stat"></span>
<div style="font-size:13px;color:#555;margin-top:4px">Kural: okunamayanı <b>?</b> ile işaretle, tahmin etme;
kırmızı sayı/not satırlarını <b>[kırmızı]</b> ile başlat. Boş satır GT'ye girmez. Taslak bu tarayıcıda otomatik saklanır.</div></div>
{items}
<script>
const KEY='nfs_transkript_'+location.pathname;
const ta=[...document.querySelectorAll('textarea')];
let draft={{}}; try{{draft=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{}}
ta.forEach(t=>{{const k=t.parentElement.dataset.key; if(draft[k]) t.value=draft[k];
  t.addEventListener('input',()=>{{draft[k]=t.value; try{{localStorage.setItem(KEY,JSON.stringify(draft))}}catch(e){{}} stat();}});}});
function stat(){{const n=ta.filter(t=>t.value.trim()).length; document.getElementById('stat').textContent=n+' / '+ta.length+' satır dolu';}}
stat();
// küçük kırpmaları okunur boya büyüt (yükseklik en az 160px, genişlik taşmasın)
document.querySelectorAll('img').forEach(im=>{{im.addEventListener('load',()=>{{
  const k=Math.max(1,160/im.naturalHeight); im.style.width=Math.min(im.parentElement.clientWidth-16, im.naturalWidth*k)+'px';}});}});
document.getElementById('hs').onchange=e=>document.body.classList.toggle('hideshort',e.target.checked);
document.getElementById('dl').onclick=()=>{{
  const rows=[{json.dumps([[r["page"], r["idx"], *r["box"], r["baseline_px"], int(r["baseline_px"] < a.short_px), r["img"]] for r in rows], ensure_ascii=False)}][0];
  let out='page\\tidx\\tx0\\ty0\\tx1\\ty1\\tbaseline_px\\tshort\\timg\\tgt\\n';
  rows.forEach((r,i)=>{{out+=r.join('\\t')+'\\t'+(ta[i].value.replace(/[\\t\\n]/g,' ').trim())+'\\n';}});
  const b=new Blob([out],{{type:'text/tab-separated-values;charset=utf-8'}});
  const a=document.createElement('a'); a.href=URL.createObjectURL(b); a.download='lines.tsv'; a.click();}};
document.getElementById('clr').onclick=()=>{{if(confirm('Taslak silinsin mi?')){{localStorage.removeItem(KEY); ta.forEach(t=>t.value=''); stat();}}}};
</script></body></html>"""
    with open(os.path.join(a.out, "sheet.html"), "w", encoding="utf8") as f:
        f.write(sheet)
    print(f"{len(rows)} satır → {a.out}/lines.tsv, sheet.html")


if __name__ == "__main__":
    main()
