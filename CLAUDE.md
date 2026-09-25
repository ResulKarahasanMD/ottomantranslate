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

## Dikkat
- `requirements.txt` **sürümleri bilerek sabitlenmiş**. Özellikle `scikit-learn==1.8.0`: `models/` altındaki pickle'lar (`2L_NN.sav` vb.) bu sürümle eğitildi. Sürümü yükseltirsen `InconsistentVersionWarning` ve sessizce hatalı tahmin alırsın — yükseltiyorsan modelleri yeniden eğit.
- `Dataset/`, `htr_models/`, `models/` ham veri ve eğitilmiş ağırlık: **değiştirme, silme.** Boyut (2026-09-25 ölçümü): proje 317 MB, ayrıca `venv/` 641 MB ve `.git` 50 MB. `htr_models/`'in 178 MB'ı NFS.d. 12369 kırpma PNG'leri; GGUF HTR modelleri (4,4 GB) 2026-08-23'te silindi, eski "7.2 GB" o dönemin ölçümüydü.
- Aktif dal `fix/ocr-robustness-and-archive-mode`. Upstream başka birine ait; push etmeden önce sor.
- OCR kalitesini smoothing/crop ile "iyileştirip" hatayı gizleme — hatalı tanınan karakteri raporla.
- Transliterasyon/çeviri çıktısında uydurma kelime tamamlama yapma; okunamayanı okunamadı olarak işaretle.
