# OCR.py

import cv2 as cv
import os
import sys
import time
from tqdm import tqdm
from glob import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
from concurrent.futures.process import BrokenProcessPool
from ocr_utils import run2, init_worker, load_model
from segmentation import extract_words
from preprocessing import binarize
from archive_preprocessing import preprocess_archive

model_name = '2L_NN.sav'

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def run(file_path, archive_mode=False, binarization="otsu",
        write_output=True, word_timeout=600, max_word_width_ratio=0.5,
        max_ink_ratio=0.3, **bin_kwargs):
    """Bir görüntü üzerinde OCR çalıştırır.

    Args:
        file_path (str): Görüntü dosyasının yolu.
        archive_mode (bool): True ise arşiv belgesi ön işlemesi (kırmızı
            mürekkep temizleme + sütun bölgelerine ayırma) uygulanır.
        binarization (str): 'otsu' (global eşik, varsayılan) veya 'sauvola'
            (yerel eşik; lekeli / düzensiz aydınlatılmış sayfalar için).
        write_output (bool): True ise sonuç output/text/<ad>.txt'e yazılır.
            Değerlendirme koşularında False verilerek kirlilik önlenir.
        word_timeout (float): Tüm kelimelerin tanınması için toplam süre
            sınırı (sn); aşılırsa sayfa hata döner, süreç asılı kalmaz.
        max_word_width_ratio (float): Bölge genişliğine oranla bundan geniş
            "kelime" segmentleri atlanır (ikilileştirme hatası koruması).
        max_ink_ratio (float): İkili görüntüde metin sayılan piksel oranı bunu
            aşarsa bölge atlanır: eşik zemini mürekkep saymıştır (temiz sayfa
            ~0.01-0.1; global Otsu gölgeli sayfada ~0.4). Karakter
            segmentasyonu böyle bir blokta dakikalar sürer.
        **bin_kwargs: İkilileştirme yöntemine özel parametreler
            (sauvola: window_size, k).

    Returns:
        tuple: (görüntü adı, karakter sayısı, kelime sayısı, tahmin metni).
            Hata durumunda (None, 0, 0, '').
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
        gray = cv.cvtColor(region, cv.COLOR_BGR2GRAY) if region.ndim == 3 else region
        ink = (binarize(gray, binarization, **bin_kwargs) > 0).mean()
        if ink > max_ink_ratio:
            print(f"Uyarı: bölge atlandı, ikilileştirme ({binarization}) piksellerin "
                  f"%{ink * 100:.0f}'ini metin saydı (sınır %{max_ink_ratio * 100:.0f}); "
                  f"zemin gölgeli/lekeli olabilir, Sauvola deneyin.")
            continue
        region_words = extract_words(region, binarization=binarization, **bin_kwargs)
        if region_words:
            words.extend(region_words)

    if not words:
        print("Error: Kelimeleri ayırmada hata oluştu.")
        return None, 0, 0, ''

    # Segmentasyon başarısızlığı koruması: bölge genişliğinin yarısından geniş
    # bir "kelime" gerçek kelime değil, ikilileştirmenin zemini metin saydığı
    # bloktur (ör. global Otsu + gölgeli sayfa). Böyle bir bloğu karakterlere
    # bölmek dakikalar sürer ve işçi süreçleri öldürür. Atlanır ve raporlanır;
    # çıktı o bölge için boş kalır (gizlenmez).
    kept, skipped = [], 0
    for w, line in words:
        if w.shape[1] > max_word_width_ratio * line.shape[1]:
            skipped += 1
        else:
            kept.append((w, line))
    if skipped:
        print(f"Uyarı: {skipped} kelime bölge genişliğinin "
              f"%{int(max_word_width_ratio * 100)}'inden geniş olduğu için "
              f"atlandı (ikilileştirme zemini metin saydı; Sauvola deneyin).")
    words = kept
    if not words:
        print("Error: Tüm kelimeler segmentasyon koruması tarafından atlandı.")
        return None, 0, 0, ''

    # Her worker modeli bir kez yükler (init_worker), her kelimede yeniden değil.
    # multiprocessing.Pool.map bir işçi ölünce sonsuza dek bekler; executor
    # BrokenProcessPool/TimeoutError ile hatayı yüzeye çıkarır.
    try:
        with ProcessPoolExecutor(max_workers=mp.cpu_count(),
                                 initializer=init_worker) as ex:
            predicted_words = list(ex.map(run2, words, timeout=word_timeout))
    except BrokenProcessPool:
        print("Error: OCR işçi süreci çöktü (bellek?); sayfa atlandı.")
        return None, 0, 0, ''
    except FutTimeout:
        print(f"Error: OCR {word_timeout} sn içinde bitmedi; sayfa atlandı.")
        return None, 0, 0, ''

    predicted_text = ' '.join(predicted_words)
    # Karakter sayisi kelime aralarina eklenen bosluklari icermez.
    char_count = sum(len(w) for w in predicted_words)

    if write_output:
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
