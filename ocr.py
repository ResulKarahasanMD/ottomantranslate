# OCR.py

import cv2 as cv
import os
import sys
import time
from tqdm import tqdm
from glob import glob
import multiprocessing as mp
from ocr_utils import run2, init_worker, load_model
from segmentation import extract_words
from archive_preprocessing import preprocess_archive

model_name = '2L_NN.sav'

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def run(file_path, archive_mode=False):
    """Bir görüntü üzerinde OCR çalıştırır.

    Args:
        file_path (str): Görüntü dosyasının yolu.
        archive_mode (bool): True ise arşiv belgesi ön işlemesi (kırmızı
            mürekkep temizleme + sütun bölgelerine ayırma) uygulanır.
    """
    full_image = cv.imread(file_path)
    img_name = os.path.basename(file_path).split('.')[0]

    if full_image is None:
        print(f"Error: Dosya bulunamadı: {file_path}")
        return None, 0, 0, ''

    # Havuz acilmadan once model dogrulanir: yoksa init_worker her worker'da
    # None doner ve tum kelimeler sessizce bos cikardi.
    if load_model() is None:
        print("Error: Eğitilmiş model bulunamadı (models/2L_NN.sav). "
              "Önce 'Model Eğit' menüsünden model eğitin.")
        return None, 0, 0, ''

    # Arşiv modu: kırmızı mürekkebi temizle ve sütunlara ayır, her bölgeden
    # ayrı ayrı kelime çıkar.
    if archive_mode:
        regions = preprocess_archive(full_image, do_split=True)
    else:
        regions = [full_image]

    words = []
    for region in regions:
        region_words = extract_words(region)
        if region_words:
            words.extend(region_words)

    if not words:
        print("Error: Kelimeleri ayırmada hata oluştu.")
        return None, 0, 0, ''

    # Her worker modeli bir kez yükler (init_worker), her kelimede yeniden değil.
    with mp.Pool(mp.cpu_count(), initializer=init_worker) as pool:
        predicted_words = pool.map(run2, words)

    predicted_text = ' '.join(predicted_words)
    # Karakter sayisi kelime aralarina eklenen bosluklari icermez.
    char_count = sum(len(w) for w in predicted_words)

    output_dir = os.path.join(_MODULE_DIR, 'output', 'text')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{img_name}.txt'), 'w', encoding='utf8') as fo:
        fo.write(predicted_text)

    return img_name, char_count, len(words), predicted_text



if __name__ == "__main__":
    image_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_MODULE_DIR, 'src', 'test')

    types = ['png', 'jpg', 'bmp']
    images_paths = []
    for t in types:
        images_paths.extend(glob(os.path.join(image_dir, f'*.{t}')))

    if not images_paths:
        print(f"Uyarı: '{image_dir}' altında görüntü bulunamadı. "
              f"Kullanım: python ocr.py <görüntü_dizini>")
        sys.exit(1)

    before = time.time()
    results = []
    for image_path in tqdm(images_paths, total=len(images_paths)):
        results.append(run(image_path))

    ok = [r for r in results if r[0] is not None]
    ok.sort(key=lambda r: r[0])

    os.makedirs(os.path.join(_MODULE_DIR, 'output'), exist_ok=True)
    with open(os.path.join(_MODULE_DIR, 'output', 'running_time.txt'), 'w') as r:
        for name, char_count, word_count, _ in ok:
            r.write(f'image#{name}: {char_count} characters in {word_count} words\n')

    after = time.time()
    print(f'{len(ok)}/{len(images_paths)} görüntü işlendi, süre: {after - before:.1f} sn')
