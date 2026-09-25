# CLAUDE.md — ottomantranslate

## Ne bu
Osmanlıca belge OCR + çeviri boru hattı. Streamlit arayüzü `app.py`. Fork: `hakantrkgl/ottomantranslate`.

| Modül | İş |
|---|---|
| `preprocessing.py`, `archive_preprocessing.py` | Görüntü hazırlama, arşiv belgesi kipi |
| `segmentation.py`, `character_segmentation.py` | Satır/karakter ayırma |
| `feature_extraction.py`, `ocr.py`, `ocr_utils.py` | Öznitelik + tanıma |
| `train.py`, `dataset.py`, `ottomanDataset.py` | Model eğitimi, `Dataset/chars/` sayımı |
| `nmt/` | OpenNMT-py tabanlı çeviri (opsiyonel menü) |

## Komutlar
```bash
source venv/bin/activate
streamlit run app.py
venv/bin/python -m unittest tests.test_binarize_evaluate -v   # hızlı birim testleri
```

## İkilileştirme ve değerlendirme
- `preprocessing.binarize(gray, method)`: `otsu` (global, varsayılan) | `sauvola`
  (yerel eşik + speckle filtresi; `window_size=31`, `k=0.3`). `ocr.run(...,
  binarization=)` ve Streamlit "İkilileştirme yöntemi" seçicisi aynı yolu kullanır.
  Sararmış/gölgeli/lekeli arşiv sayfasında Sauvola; temiz taramada ikisi eşdeğer.
- `ocr.run` dev "kelime" bloklarını (bölge genişliğinin >%50'si) atlar ve uyarır;
  tanıma `ProcessPoolExecutor` + zaman aşımı ile çalışır, işçi ölünce asılmaz.
- **Depoda karakter düzeyinde ground truth YOK.** `output/text/NFS_*` transkriptleri
  Latin çeviriyazı + `[?]` işaretli en-iyi-çaba okumalardır, CER için kullanılmaz.
- Sentetik GT: `evaluation/synthetic_lines.txt` (28 harf, Osmanlıca) →
  `tools/render_lines` (Swift/CoreText; PIL'de raqm yok) → `tools/degrade.py`
  (clean/gradient/stain/lowcontrast/combo) → `evaluate.py --gt-dir … --arabic-norm`.
  Rapor `out_<tarih>_cer/cer_report.md`. Yeniden üretim:
  ```bash
  xcrun swiftc -O -o tools/render_lines tools/render_lines.swift
  tools/render_lines evaluation/synthetic_lines.txt evaluation/synthetic_clean DecoTypeNaskh 44 6
  venv/bin/python tools/degrade.py evaluation/synthetic_clean evaluation/synthetic
  venv/bin/python evaluate.py --gt-dir evaluation/synthetic --arabic-norm
  ```
  `evaluate.py` dosyadan çalıştırılır (heredoc/stdin ile multiprocessing spawn tuzağı).

## kraken (ayrı venv, 2026-09-26)
- Kurulum: `~/projects/venvs/kraken` (uv, py3.12, kraken 7.1.1, torch 2.14 MPS). Modeller
  `~/projects/venvs/kraken_models/`: `ottoman_best.mlmodel` ve `arabic_best.mlmodel`
  (OpenITI Printed Base, zenodo 7050342 / 7050296), `seg_general.mlmodel` (çok yazılı
  baseline segmentasyonu, zenodo 14602569). Depo dışı; `kraken get <DOI>` ile yeniden iner.
- **Şart:** `ocr --base-dir R`; varsayılan BiDi yeniden sıralama çıktıyı tersine çevirir.
  Model Farsça ک/ی üretir; CER öncesi ك/ي'ye normalize edilir (`tools/kraken_eval.py`).
- Matbu/sentetik sayfa: **Sauvola ön-ikili → `segment -x -d horizontal-rl` (kutu) → ocr**
  micro CER 0,016 (10 sayfa); kraken'in kendi nlbin'i 0,229 (düşük kontrastta çöker);
  ham gri + blla baseline 0,830 (satırları parçalıyor). Rapor `evaluation/kraken_2026-09-26.md`.
- El yazısı NFS sayfası: `seg_general` taban çizgileri iyi (138 satır, kırmızı sayılar
  atlanıyor), ama Printed Ottoman modeli metni okuyamıyor; el yazısı için satır GT + ince
  ayar gerekir (`ketos train`).
- **Satır GT hazırlığı:** `tools/nfs_lines_export.py` → `out_20260926_nfs_lines/` (9 sayfa,
  745 satır kırpması, `lines.tsv` boş `gt` sütunu, `sheet.html` RTL transkripsiyon sayfası;
  `short=1` olan 209 satır kırmızı sayı/parça adayı). Sayfa `python3 -m http.server 8765
  --directory out_20260926_nfs_lines` ile açılır (`.claude/launch.json`: nfs-lines-sheet);
  `file://` ile açılınca kırpmalar yüklenmez. Sayfa 00000-00001 arşiv kapağı (Latin), GT'ye girmez.
  Doldurulan TSV "TSV indir" ile alınır; okunamayan satır `?`, kırmızı not `[kırmızı] …`.

## Dikkat
- `requirements.txt` **sürümleri bilerek sabitlenmiş**. Özellikle `scikit-learn==1.8.0`: `models/` altındaki pickle'lar (`2L_NN.sav` vb.) bu sürümle eğitildi. Sürümü yükseltirsen `InconsistentVersionWarning` ve sessizce hatalı tahmin alırsın — yükseltiyorsan modelleri yeniden eğit.
- `Dataset/`, `htr_models/`, `models/` ham veri ve eğitilmiş ağırlık: **değiştirme, silme.** Boyut (2026-09-25 ölçümü): proje 317 MB, ayrıca `venv/` 641 MB ve `.git` 50 MB. `htr_models/`'in 178 MB'ı NFS.d. 12369 kırpma PNG'leri; GGUF HTR modelleri (4,4 GB) 2026-08-23'te silindi, eski "7.2 GB" o dönemin ölçümüydü.
- Aktif dal `fix/ocr-robustness-and-archive-mode`. Upstream başka birine ait; push etmeden önce sor.
- OCR kalitesini smoothing/crop ile "iyileştirip" hatayı gizleme — hatalı tanınan karakteri raporla.
- Transliterasyon/çeviri çıktısında uydurma kelime tamamlama yapma; okunamayanı okunamadı olarak işaretle.
