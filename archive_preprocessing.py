"""Arşiv belgesi (BOA / nüfus defteri vb.) ön işleme yardımcıları.

Bu modül, klasik segmentasyon tabanlı OCR hattının varsaydığı 'temiz, tek
renkli, tek sütunlu' koşulu sağlamayan belgeler için giriş kalitesini
iyileştirmeyi hedefler. İki ana işlem sağlar:

1. remove_red_ink : Kırmızı mürekkeple yazılmış düzeltme/iptal (üstü çizik)
   işaretlerini ve kırmızı ek notları temizler; yalnızca siyah metni bırakır.
   Bu, siyah karakterlerin kırmızı çapraz çizgilerle birleşip segmentasyonu
   bozmasını önler.

2. split_columns : İki (veya daha fazla) sütunlu düzeni dikey boşluk
   analizine göre ayrı bölgelere böler; her sütun bağımsız işlenebilir.

NOT: Bu ön işleme, çapraz/çok açılı el yazısını okuyabilen bir OCR modeli
yerine geçmez; yalnızca mevcut hattın girdisini temizler. Gerçek arşiv OCR'ı
için segmentasyonsuz (CRNN/Transformer) bir modele geçiş gerekir.
"""

import numpy as np
import cv2 as cv


def remove_red_ink(bgr_image, whiten=True, hue_max=10, s_min=45):
    """Kırmızı mürekkebi görüntüden kaldırır.

    HSV uzayında kırmızı tonlarını maskeler ve bu pikselleri kağıt rengine
    (beyaz) çeker; böylece geriye yalnızca siyah metin kalır.

    Eşikler NFS.d. 12369 s.2 üzerinde ölçüldü (2026-09-26): kırmızı mürekkep
    hue 0-8 / 170-180, S≈110; sepya siyah mürekkep hue 8-25, S≈55; kağıt
    hue≈16, S≈27. Eski hue üst sınırı 15 siyah bloğun koyu piksellerinin
    %32'sini yutuyordu (vuruş içinde beyaz delikler); 10'a çekince %14'e
    düştü, kırmızı temizliği %92,5'te sabit kaldı. S eşiğini yükseltmek
    kırmızıyı kaçırıyor (S≥70 → %74), o yüzden 45'te bırakıldı.

    Args:
        bgr_image (numpy.ndarray): BGR (OpenCV) renkli giriş görüntüsü.
        whiten (bool): True ise kırmızı pikseller beyaza boyanır.
        hue_max (int): Kırmızı sayılan hue yarı-genişliği (0..hue_max ve
            180-hue_max..180). Sepya siyah mürekkep 8'in üzerinde başlar.
        s_min (int): Minimum doygunluk; siyah mürekkep S≈0-45 civarındadır.

    Returns:
        numpy.ndarray: Kırmızı mürekkebi temizlenmiş BGR görüntü.
    """
    if bgr_image is None or len(bgr_image.shape) != 3:
        return bgr_image

    hsv = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)

    # Kırmızı, HSV'de sarmalanma (wrap-around) nedeniyle iki aralıkta bulunur.
    # Doygunluk tabanı (S>=45) hem solmuş/koyu (bordo) arşiv mürekkebini yakalar
    # hem de düşük doygunluklu siyah metni korur (siyah S~0).
    lower1 = np.array([0, s_min, 40])
    upper1 = np.array([hue_max, 255, 255])
    lower2 = np.array([180 - hue_max, s_min, 40])
    upper2 = np.array([180, 255, 255])

    mask = cv.inRange(hsv, lower1, upper1) | cv.inRange(hsv, lower2, upper2)

    # Maskeyi biraz genişlet ki kırmızı çizgilerin kenar pikselleri de temizlensin.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.dilate(mask, kernel, iterations=1)

    out = bgr_image.copy()
    if whiten:
        out[mask > 0] = (255, 255, 255)
    return out


def _vertical_ink_profile(gray):
    """Sütun (dikey şerit) başına mürekkep yoğunluğu profili."""
    # Metin koyu (düşük değer) -> ters çevirip topla
    inv = 255 - gray
    return inv.sum(axis=0).astype(np.float64)


def split_columns(bgr_image, min_gap_ratio=0.04, min_col_ratio=0.12):
    """Çok sütunlu bir sayfayı dikey boşluklara göre sütun bölgelerine ayırır.

    Args:
        bgr_image (numpy.ndarray): BGR giriş görüntüsü.
        min_gap_ratio (float): Sütun ayıracı sayılacak minimum boşluk genişliği
            (sayfa genişliğine oran).
        min_col_ratio (float): Geçerli sayılacak minimum sütun genişliği oranı.

    Returns:
        list[tuple[int, int]]: (x_başlangıç, x_bitiş) sütun sınırlarının listesi.
            Tek sütun bulunursa tüm genişliği kapsayan tek bir aralık döner.
    """
    if bgr_image is None or len(bgr_image.shape) != 3:
        return [(0, bgr_image.shape[1])] if bgr_image is not None else []

    gray = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)
    w = gray.shape[1]
    profile = _vertical_ink_profile(gray)

    # Yumuşat
    k = max(3, int(w * 0.01) | 1)
    profile = cv.GaussianBlur(profile.reshape(1, -1), (k, 1), 0).ravel()

    # Boşluk eşiği: profilin düşük yüzdelik dilimi
    threshold = np.percentile(profile, 20)
    is_gap = profile <= threshold

    min_gap = int(w * min_gap_ratio)
    min_col = int(w * min_col_ratio)

    # Metin (boşluk olmayan) bölgelerini bul
    cols = []
    start = None
    gap_run = 0
    for x in range(w):
        if not is_gap[x]:
            if start is None:
                start = x
            gap_run = 0
        else:
            if start is not None:
                gap_run += 1
                if gap_run >= min_gap:
                    end = x - gap_run + 1
                    if end - start >= min_col:
                        cols.append((start, end))
                    start = None
                    gap_run = 0
    if start is not None and (w - start) >= min_col:
        cols.append((start, w))

    if not cols:
        return [(0, w)]
    return cols


def preprocess_archive(bgr_image, do_split=True):
    """Arşiv belgesi için tam ön işleme.

    Kırmızı mürekkebi temizler ve (istenirse) sütun bölgelerine ayırır.

    Args:
        bgr_image (numpy.ndarray): BGR giriş görüntüsü.
        do_split (bool): True ise sütunlara ayrılmış bölge görüntüleri döner.

    Returns:
        list[numpy.ndarray]: İşlenmiş bölge görüntülerinin listesi (BGR),
            belge okuma sırasında (en sağdaki sütun önce). Bölme kapalıysa
            tek elemanlı liste döner.
    """
    cleaned = remove_red_ink(bgr_image)
    if not do_split:
        return [cleaned]

    # split_columns sütunları soldan sağa verir; Osmanlıca/Arapça belgede
    # okuma sırası sağdan sola olduğu için bölgeler ters çevrilerek döndürülür.
    regions = []
    for (x0, x1) in reversed(split_columns(cleaned)):
        regions.append(cleaned[:, x0:x1])
    return regions if regions else [cleaned]
