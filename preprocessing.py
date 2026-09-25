import numpy as np  # NumPy kütüphanesi
import cv2 as cv  # OpenCV kütüphanesi
# scipy.ndimage.interpolation alt modulu SciPy 1.10'da kaldirildi; rotate
# artik dogrudan scipy.ndimage'dan gelir.
from scipy.ndimage import rotate as ndi_rotate
from PIL import Image as im  # PIL kütüphanesi


def binary_otsus(image, filter:int=1):
    """ Otsu'nun ikili (siyah-beyaz) resim dönüştürme yöntemini uygular.

    Args:
        image (numpy.ndarray): Giriş görüntü.
        filter (int): Görüntüyü yumuşatma (blurring) için kullanılan filtre boyutu. Varsayılan olarak 1.

    Returns:
        numpy.ndarray: Otsu'nun ikili dönüşüm sonucu elde edilen görüntü."""

    # Giriş resmi gri tonlamaya dönüştürme
    if len(image.shape) == 3:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Otsu'nun ikili dönüşümü
    if filter != 0:
        blur = cv.GaussianBlur(gray_img, (3,3), 0)
        binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    else:
        binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    
    # Morphological Opening
    # kernel = np.ones((3,3),np.uint8)
    # clean_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

    return binary_img


# Desteklenen ikilileştirme yöntemleri (segmentation.preprocess ve ocr.run
# tarafından paylaşılır).
BINARIZATION_METHODS = ("otsu", "sauvola")


def binary_sauvola(gray_img, window_size: int = 31, k: float = 0.3,
                   min_area: int = None):
    """Sauvola yerel eşikleme ile ikilileştirir.

    Otsu tek bir global eşik seçer; sararmış, lekeli veya düzensiz aydınlatılmış
    arşiv sayfalarında bu eşik ya lekeleri metin sayar ya da soluk vuruşları
    yutar. Sauvola her piksel için komşuluğun ortalama ve standart sapmasından
    ayrı bir eşik hesaplar: T = m * (1 + k * (s / R - 1)).

    Args:
        gray_img (numpy.ndarray): Koyu metin / açık zemin gri görüntü
            (ters ÇEVRİLMEMİŞ). Renkli verilirse griye çevrilir.
        window_size (int): Yerel pencere kenarı (tek sayı olmalı; çift
            verilirse 1 artırılır). Vuruş kalınlığının ~3-5 katı iyi çalışır.
        k (float): Sauvola k parametresi (0.2-0.5 tipik). Büyük k daha az
            piksel'i metin sayar (daha ince/temiz), küçük k daha fazlasını.
            Varsayılan 0.3, sentetik bozulma taramasında (2026-09-26) tüm
            koşullarda satır sayısını koruyan en küçük değerdi; 0.2 gürültüyü
            metin sayıyor, 0.5 soluk vuruşları parçalıyordu.
        min_area (int): Bu piksel alanından küçük bağlı bileşenler (speckle)
            atılır. None ise (window_size // 8) ** 2, en az 4 px. Harf
            noktalarından küçük kalmalıdır; düşük çözünürlükte küçültün.

    Returns:
        numpy.ndarray: uint8 ikili görüntü; metin 255 (beyaz), zemin 0.
            Polarite binary_otsus(ters çevrilmiş gri) ile aynıdır, böylece
            segmentasyon hattına doğrudan verilebilir.
    """
    # skimage yalnızca burada gerekir; modül import'unu ağırlaştırmamak için
    # fonksiyon içinde alınır.
    from skimage.filters import threshold_sauvola

    if len(gray_img.shape) == 3:
        gray_img = cv.cvtColor(gray_img, cv.COLOR_BGR2GRAY)

    window_size = int(window_size)
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1

    thresh = threshold_sauvola(gray_img, window_size=window_size, k=k)
    # Koyu metin: yerel eşiğin altındaki pikseller mürekkeptir.
    binary = (gray_img < thresh).astype(np.uint8) * 255

    # Speckle temizliği: yerel eşik gürültülü düz zeminde tek tük piksel
    # üretir; bunlar projeksiyon segmentasyonunda sahte satır/kelime olur.
    if min_area is None:
        min_area = max(4, (window_size // 8) ** 2)
    if min_area > 0:
        n, labels, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
        if n > 1:
            keep = np.zeros(n, dtype=bool)
            keep[1:] = stats[1:, cv.CC_STAT_AREA] >= min_area
            binary = keep[labels].astype(np.uint8) * 255
    return binary


def binarize(gray_img, method: str = "otsu", **kwargs):
    """Yöntem adına göre ikilileştirme uygular.

    Args:
        gray_img (numpy.ndarray): Koyu metin / açık zemin gri görüntü.
        method (str): 'otsu' veya 'sauvola'.
        **kwargs: Yönteme özel parametreler (sauvola: window_size, k).

    Returns:
        numpy.ndarray: uint8 ikili görüntü, metin 255 / zemin 0.
    """
    method = (method or "otsu").lower()
    if method == "otsu":
        # Mevcut hat: griyi ters çevirip Otsu; metin beyaz kalır.
        inverted = cv.bitwise_not(gray_img)
        return binary_otsus(inverted, kwargs.get("filter", 0))
    if method == "sauvola":
        return binary_sauvola(gray_img,
                              window_size=kwargs.get("window_size", 31),
                              k=kwargs.get("k", 0.3),
                              min_area=kwargs.get("min_area"))
    raise ValueError(
        f"Bilinmeyen ikilileştirme yöntemi: {method!r}; "
        f"geçerli: {BINARIZATION_METHODS}")


def find_score(arr, angle):

    """Belirtilen açıda döndürülmüş görüntünün skorunu hesaplar.

    Args:
        arr (numpy.ndarray): Giriş görüntü.
        angle (float): Açı.

    Returns:
        numpy.ndarray: Histogram.
        float: Skor."""
    data = ndi_rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def deskew(binary_img):
    """Görüntüyü doğrultma işlemi yapar.

    Args:
        binary_img (numpy.ndarray): İkili (siyah-beyaz) görüntü.

    Returns:
        numpy.ndarray: Doğrultulmuş görüntü."""
    
    ht, wd = binary_img.shape
    # _, binary_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    # pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = (binary_img // 255.0)  # Görüntüyü ikili (siyah-beyaz) hale getirme

    delta = 0.1
    limit = 3
    angles = np.arange(-limit, limit+delta, delta)
    scores = []

     # Farklı açılarda skorları hesaplama
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # print('Best angle: {}'.formate(best_angle))

    # Doğrultma işlemini gerçekleştirme
    data = ndi_rotate(bin_img, best_angle, reshape=False, order=0)
    img = im.fromarray((255 * data).astype("uint8"))

    # img.save('skew_corrected.png')
    pix = np.array(img)
    return pix

def vexpand(gray_img, color:int):
    """Gri tonlamalı görüntüyü dikey olarak genişletir.

    Args:
        gray_img (numpy.ndarray): Gri tonlamalı giriş görüntüsü.
        color (int): Genişletilen bölgenin rengi (siyah için 0, beyaz için 1).

    Returns:
        numpy.ndarray: Genişletilmiş görüntü."""

    color = 1 if color > 0 else 0
    (h, w) = gray_img.shape[:2]
    space = np.ones((10, w)) * 255 * color

    return np.block([[space], [gray_img], [space]])


def hexpand(gray_img, color:int):
    """Gri tonlamalı görüntüyü yatay olarak genişletir.

    Args:
        gray_img (numpy.ndarray): Gri tonlamalı giriş görüntüsü.
        color (int): Genişletilen bölgenin rengi (siyah için 0, beyaz için 1).

    Returns:
        numpy.ndarray: Genişletilmiş görüntü."""

    color = 1 if color > 0 else 0
    (h, w) = gray_img.shape[:2]
    space = np.ones((h, 10)) * 255 * color

    return np.block([space, gray_img, space])

def valid(row, col, vis, word):
    """Verilen satır ve sütun indeksleri için geçerli bir hücre kontrolü yapar.

    Args:
        row (int): Satır indeksi.
        col (int): Sütun indeksi.
        vis (numpy.ndarray): Ziyaret edilen hücrelerin izdüşümü.
        word (numpy.ndarray): Kelimenin (görüntünün) piksel değerlerini içeren dizi.

    Returns:
        bool: Hücre geçerli ise True, değilse False döner."""

    return (row < vis.shape[0] and col < vis.shape[1] and row >= 0 and col >=0 and vis[row][col] == 0 and word[row][col] > 0)

def dfs(row, col, vis, word):
    """ Derin öncelikli arama (DFS) algoritması ile bağlı bileşenleri bulur.

    Args:
        row (int): Başlangıç satır indeksi.
        col (int): Başlangıç sütun indeksi.
        vis (numpy.ndarray): Ziyaret edilen hücrelerin izdüşümü.
        word (numpy.ndarray): Kelimenin (görüntünün) piksel değerlerini içeren dizi.

    Returns:
        None"""

    dX = [0,0,1,1,-1,-1,1,-1]
    dY = [1,-1,0,1,0,-1,-1,1]
    vis[row][col] += 1
    for i in range(8):
        if(valid(row+dX[i],col+dY[i],vis, word)):
            dfs(row+dX[i], col+dY[i], vis, word)
    return
