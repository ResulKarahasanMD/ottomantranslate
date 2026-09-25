"""Osmanlıca HTR (Handwritten Text Recognition) POC.

Hazır, önceden eğitilmiş bir vision-language modeli (Ottoman OCR Qwen2.5-VL,
GGUF 4-bit kuantize) ile Osmanlıca belge görüntülerini doğrudan okur. Klasik
karakter-segmentasyon hattının aksine, sayfayı/satırı bütün alır; bu yüzden
çapraz/yoğun arşiv yazısında dahi çökme yaşamaz.

Gereksinimler:
    - llama.cpp'nin multimodal CLI'si: `llama-mtmd-cli` (brew install llama.cpp)
    - Model dosyaları (htr_models/ altında):
        ottoman-q4km.gguf      (kuantize model ağırlıkları)
        ottoman-mmproj.gguf    (vision projektörü)

Kullanım:
    python htr_poc.py path/to/image.png
    python htr_poc.py path/to/image.png --archive   # kırmızı mürekkebi temizle

Not: Bu bir POC'tur. Model yalnızca ~6.900 örnekle LoRA fine-tune edilmiştir ve
yoğun siyakat/rika nüfus defterlerinde sınırlı doğruluk verir. Üretim kalitesi
için bu belge tipinden transkribe edilmiş veriyle yeniden eğitim gerekir.
"""

import os
import sys
import subprocess
import tempfile

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODULE_DIR, "htr_models", "ottoman-q4km.gguf")
MMPROJ_PATH = os.path.join(MODULE_DIR, "htr_models", "ottoman-mmproj.gguf")

SYSTEM_PROMPT = (
    "You are an expert paleographer specializing in Ottoman Turkish documents "
    "written in Arabic script. Your task is to transcribe Ottoman Turkish "
    "handwritten or printed text lines into Modern Turkish."
)
USER_PROMPT = "Bu Osmanlı Türkçesi metni Modern Türkçeye transkribe et."


def _maybe_clean_red(image_path):
    """Arşiv modu: kırmızı mürekkebi temizleyip geçici bir dosyaya yazar."""
    import cv2 as cv
    from archive_preprocessing import remove_red_ink

    img = cv.imread(image_path)
    if img is None:
        return image_path, None
    cleaned = remove_red_ink(img)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv.imwrite(tmp.name, cleaned)
    return tmp.name, tmp.name


def transcribe(image_path, archive_mode=False, n_predict=512, cli="llama-mtmd-cli"):
    """Bir görüntüyü Osmanlıca HTR modeliyle transkribe eder ve metni döndürür."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MMPROJ_PATH):
        raise FileNotFoundError(
            f"Model dosyaları bulunamadı:\n  {MODEL_PATH}\n  {MMPROJ_PATH}\n"
            "Önce htr_models/ altına GGUF + mmproj dosyalarını indirin."
        )

    tmp_to_delete = None
    if archive_mode:
        image_path, tmp_to_delete = _maybe_clean_red(image_path)

    cmd = [
        cli,
        "-m", MODEL_PATH,
        "--mmproj", MMPROJ_PATH,
        "--image", image_path,
        "-sys", SYSTEM_PROMPT,
        "-p", USER_PROMPT,
        "-n", str(n_predict),
        "--temp", "0.1",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    finally:
        if tmp_to_delete and os.path.exists(tmp_to_delete):
            os.unlink(tmp_to_delete)

    if proc.returncode != 0:
        raise RuntimeError(f"llama-mtmd-cli hata verdi:\n{proc.stderr[-2000:]}")

    return proc.stdout.strip()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--archive"]
    archive = "--archive" in sys.argv
    if not args:
        print("Kullanım: python htr_poc.py <görüntü> [--archive]")
        sys.exit(1)
    print(transcribe(args[0], archive_mode=archive))
