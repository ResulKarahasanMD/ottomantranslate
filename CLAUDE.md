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
```

## Dikkat
- `requirements.txt` **sürümleri bilerek sabitlenmiş**. Özellikle `scikit-learn==1.8.0`: `models/` altındaki pickle'lar (`2L_NN.sav` vb.) bu sürümle eğitildi. Sürümü yükseltirsen `InconsistentVersionWarning` ve sessizce hatalı tahmin alırsın — yükseltiyorsan modelleri yeniden eğit.
- `Dataset/`, `htr_models/`, `models/` ham veri ve eğitilmiş ağırlık: **değiştirme, silme.** Boyut (2026-09-25 ölçümü): proje 317 MB, ayrıca `venv/` 641 MB ve `.git` 50 MB. `htr_models/`'in 178 MB'ı NFS.d. 12369 kırpma PNG'leri; GGUF HTR modelleri (4,4 GB) 2026-08-23'te silindi, eski "7.2 GB" o dönemin ölçümüydü.
- Aktif dal `fix/ocr-robustness-and-archive-mode`. Upstream başka birine ait; push etmeden önce sor.
- OCR kalitesini smoothing/crop ile "iyileştirip" hatayı gizleme — hatalı tanınan karakteri raporla.
- Transliterasyon/çeviri çıktısında uydurma kelime tamamlama yapma; okunamayanı okunamadı olarak işaretle.
