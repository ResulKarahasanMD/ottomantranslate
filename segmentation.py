import numpy as np  # NumPy kütüphanesi
import cv2 as cv  # OpenCV kütüphanesi
from preprocessing import binarize, deskew  # preprocessing.py
from utilities import projection, save_image  # utilities.py
from glob import glob  # Glob modülü, dosya aramak için kullanılır.



def preprocess(image, binarization: str = "otsu", **bin_kwargs):

    """    Görüntüyü ön işleme adımlarını uygulayan fonksiyon.

    Args:
        image (numpy.ndarray): Giriş görüntü (BGR veya gri).
        binarization (str): 'otsu' (global, varsayılan) veya 'sauvola'
            (yerel; lekeli/düzensiz aydınlatılmış arşiv sayfaları için).
        **bin_kwargs: Yönteme özel parametreler (bkz. preprocessing.binarize).

    Returns:
        numpy.ndarray: Ön işlemden geçirilmiş görüntü (metin 255, zemin 0)."""
    # Gri tonlamalı görüntüyü elde etme
    if len(image.shape) == 3:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # İkilileştirme (metin beyaz / zemin siyah polaritesi korunur)
    binary_img = binarize(gray_img, binarization, **bin_kwargs)
    # cv.imwrite('origin.png', gray_img)

     # Görüntüyü doğrultma işlemi
    deskewed_img = deskew(binary_img)
    # cv.imwrite('output.png', deskewed_img)

    # binary_img = binary_otsus(deskewed_img, 0)
    # breakpoint()

    # Visualize

    # breakpoint()
    return deskewed_img


def projection_segmentation(clean_img, axis, cut=3):
    """İz düşüm tabanlı segmentasyon işlemini uygular.

    Args:
        clean_img (numpy.ndarray): Temizlenmiş görüntü.
        axis (str): İz düşümün hangi eksende yapılacağı ('horizontal' veya 'vertical').
        cut (int): Kesme eşiği.

    Returns:
        list: Segmentlenmiş görüntü parçalarının listesi."""
    segments = []
    start = -1
    cnt = 0

    projection_bins = projection(clean_img, axis)
    for idx, projection_bin in enumerate(projection_bins):

        if projection_bin != 0:
            cnt = 0
        if projection_bin != 0 and start == -1:
            start = idx
        if projection_bin == 0 and start != -1:
            cnt += 1
            if cnt >= cut:
                if axis == 'horizontal':
                    segments.append(clean_img[max(start-1, 0):idx, :])
                elif axis == 'vertical':
                    segments.append(clean_img[:, max(start-1, 0):idx])
                cnt = 0
                start = -1

    # Murekkep goruntunun kenarina kadar uzaniyorsa dongudeki flush hic
    # tetiklenmez; son segment (en alt satir / RTL'de satirin ilk kelimesi)
    # burada eklenir.
    if start != -1:
        if axis == 'horizontal':
            segments.append(clean_img[max(start-1, 0):, :])
        elif axis == 'vertical':
            segments.append(clean_img[:, max(start-1, 0):])

    return segments


# Line Segmentation

def line_horizontal_projection(image, cut=3, binarization="otsu", **bin_kwargs):

    """Yatay iz düşüm tabanlı satır segmentasyonu uygular.

    Args:
        image (numpy.ndarray): Giriş görüntü.
        cut (int): Kesme eşiği.
        binarization (str): İkilileştirme yöntemi ('otsu' | 'sauvola').
        **bin_kwargs: Yönteme özel parametreler.

    Returns:
        list: Satırların listesi."""
    
    # Giriş görüntüsünü ön işleme tabi tutma
    clean_img = preprocess(image, binarization, **bin_kwargs)


    # Segmentasyon   
    lines = projection_segmentation(clean_img, axis='horizontal', cut=cut)

    return lines


# Word Segmentation
#----------------------------------------------------------------------------------------
def word_vertical_projection(line_image, cut=3):

    """Dikey iz düşüm tabanlı kelime segmentasyonu uygular.

    Args:
        line_image (numpy.ndarray): Satır görüntüsü.
        cut (int): Kesme eşiği.

    Returns:
        list: Kelimelerin listesi."""
    
    line_words = projection_segmentation(line_image, axis='vertical', cut=cut)
    line_words.reverse()
    
    return line_words


def extract_words(img, visual=0, binarization="otsu", **bin_kwargs):

    """Görüntüden kelimeleri çıkaran fonksiyon.

    Args:
        img (numpy.ndarray): Giriş görüntü.
        visual (int): Görüntü işleme sonuçlarını görselleştirme modu.
        binarization (str): İkilileştirme yöntemi ('otsu' | 'sauvola').
        **bin_kwargs: Yönteme özel parametreler (sauvola: window_size, k).

    Returns:
        list: (Kelime, Satır) çiftlerinin listesi."""

    lines = line_horizontal_projection(img, binarization=binarization, **bin_kwargs)
    words = []
    
    for idx, line in enumerate(lines):
        
        if visual:
            save_image(line, 'lines', f'line{idx}')

        line_words = word_vertical_projection(line)
        for w in line_words:
            # if len(words) == 585:
            #     print(idx)
            words.append((w, line))
        # words.extend(line_words)

    # breakpoint()
    if visual:
        for idx, word in enumerate(words):
            save_image(word[0], 'words', f'word{idx}')

    return words


if __name__ == "__main__":
    # Örnek bir görüntü yükleniyor ve kelimeler çıkarılıyor
    img = cv.imread('../Dataset/scanned/capr196.png')
    extract_words(img, 1) # Kelimeleri çıkarırken görselleştirme modunu aktif hale getiriyoruz.