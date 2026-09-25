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
- `requirements.txt` **sürümleri bilerek sabitlenmiş**. Özellikle `scikit-learn==1.8.0`: `htr_models/` ve `models/` altındaki pickle'lar bu sürümle eğitildi. Sürümü yükseltirsen `InconsistentVersionWarning` ve sessizce hatalı tahmin alırsın — yükseltiyorsan modelleri yeniden eğit.
- `Dataset/`, `htr_models/`, `models/` ham veri ve eğitilmiş ağırlık: **değiştirme, silme.** Dizin 7.2 GB, disk dar.
- Aktif dal `fix/ocr-robustness-and-archive-mode`. Upstream başka birine ait; push etmeden önce sor.
- OCR kalitesini smoothing/crop ile "iyileştirip" hatayı gizleme — hatalı tanınan karakteri raporla.
- Transliterasyon/çeviri çıktısında uydurma kelime tamamlama yapma; okunamayanı okunamadı olarak işaretle.
