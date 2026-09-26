import numpy as np  # NumPy kütüphanesi
import cv2 as cv  # OpenCV kütüphanesi
import os  # İşletim sistemi fonksiyonları için kullanılır
import re  # Düzenli ifadeler için kullanılır
import random  # Rastgele sayılar üretmek için kullanılır
# Kendi tanımladığınız utilities modülünden fonksiyonları içe aktarıyoruz.
from utilities import projection
from glob import glob  # Dosya aramak için kullanılır
from tqdm import tqdm  # İlerleme çubuğunu oluşturmak için kullanılır

from sklearn.utils import shuffle  # Veriyi karıştırmak için kullanılır
# Veri kümesini eğitim ve test alt kümelerine bölmek için kullanılır
from sklearn.model_selection import train_test_split
from sklearn import svm  # Destek Vektör Makineleri (SVM) için kullanılır
# Yapay Sinir Ağları için kullanılır
from sklearn.neural_network import MLPClassifier
# Naive Bayes sınıflandırıcı için kullanılır
from sklearn.naive_bayes import GaussianNB
# Doğruluk skoru hesaplamak için kullanılır
from sklearn.metrics import accuracy_score
import pickle  # Nesneleri serileştirmek ve deserileştirmek için kullanılır

# Sınıflandırılacak karakterler
chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
         'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'لا', 'پ', 'چ', 'ژ', '،', 'ی']  # İşlenen karakterlerin listesi

train_ratio = 0.8  # Eğitim verisi oranı
script_path = os.getcwd()  # Kodun bulunduğu dizin

# Dataset/chars klasörünü hem proje kökünden ('Dataset/chars') hem de alt
# klasörden ('../Dataset/chars') çalıştırmaya dayanıklı şekilde bul.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _chars_base_dir():
    for candidate in (
        os.path.join(_MODULE_DIR, 'Dataset', 'chars'),
        os.path.join(os.getcwd(), 'Dataset', 'chars'),
        os.path.join(os.getcwd(), '..', 'Dataset', 'chars'),
    ):
        if os.path.isdir(candidate):
            return candidate
    # Bulunamazsa modül dizinine göre varsayılan
    return os.path.join(_MODULE_DIR, 'Dataset', 'chars')
classifiers = [svm.LinearSVC(), MLPClassifier(alpha=1e-4, hidden_layer_sizes=(100,), max_iter=1000),
               MLPClassifier(alpha=1e-5, hidden_layer_sizes=(200, 100,), max_iter=1000), GaussianNB()]  # Sınıflandırıcı modelleri


# Sınıflandırıcı model isimleri
names = ['LinearSVM', '1L_NN', '2L_NN', 'Gaussian_Naive_Bayes']
ALL_MODELS = ["LinearSVM", "1L_NN", "2L_NN", "Gaussian_Naive_Bayes"]
selected_models = []  # Seçilen model isimlerini tutacak liste (boş = hepsi)
results = []  # Egitilen modellerin (ad, skor) ciftleri

def set_selected_models(models):
    global selected_models
    selected_models = models

width = 25  # Görüntü genişliği
height = 25  # Görüntü yüksekliği
dim = (width, height)  # Yeni boyut


def bound_box(img_char):
    # İz düşüm tabanlı sınırlayıcı kutu hesaplama
    HP = projection(img_char, 'horizontal')  # Yatay iz düşüm
    VP = projection(img_char, 'vertical')  # Dikey iz düşüm

    top = -1  # Üst sınır
    down = -1  # Alt sınır
    left = -1  # Sol sınır
    right = -1  # Sağ sınır

    i = 0
    while i < len(HP):
        if HP[i] != 0:
            top = i
            break
        i += 1

    i = len(HP)-1
    while i >= 0:
        if HP[i] != 0:
            down = i
            break
        i -= 1

    i = 0
    while i < len(VP):
        if VP[i] != 0:
            left = i
            break
        i += 1

    i = len(VP)-1
    while i >= 0:
        if VP[i] != 0:
            right = i
            break
        i -= 1

    if top == -1 or left == -1:
        # Tamamen bos (murekkepsiz) parca: bos dilim cv.resize'da sessiz
        # karakter kaybina donusuyordu; acik hata cagiranin atlamasini saglar.
        raise ValueError('bound_box: murekkep icermeyen bos karakter goruntusu')

    return img_char[top:down+1, left:right+1]  # Sınırlayıcı kutuyu döndürme


def binarize(char_img):
    # Görüntüyü ikili hale getirme
    _, binary_img = cv.threshold(char_img, 127, 255, cv.THRESH_BINARY)
    binary_char = binary_img // 255  # 0 ve 1 değerlerine dönüştürme

    return binary_char  # İkili hale getirilmiş karakter görüntüsünü döndürme


def prepare_char(char_img):
    # Karakter görüntüsünü hazırlama
    binary_char = binarize(char_img)  # İkili hale getirme
    char_box = bound_box(binary_char)  # Sınırlayıcı kutu hesaplama
    # Yeniden boyutlandırma
    resized = cv.resize(char_box, dim, interpolation=cv.INTER_AREA)

    return resized  # Hazırlanmış karakter görüntüsünü döndürme


def featurizer(char_img):
    # Özellik çıkarımı
    flat_char = char_img.flatten()  # Görüntüyü düzleştirme

    return flat_char  # Düzleştirilmiş görüntüyü döndürme


def read_data(limit=4000):
    # Veri okuma işlemi
    X = []  # Girdi verisi
    Y = []  # Etiket verisi
    base_dir = _chars_base_dir()
    print(f"For each char (data dir: {base_dir})")
    for char in tqdm(chars, total=len(chars)):

        folder = os.path.join(base_dir, char)
        char_paths = glob(os.path.join(folder, '*.png'))

        if os.path.isdir(folder):
            print(f'\nReading images for char {char}')
            for char_path in tqdm(char_paths[:limit], total=len(char_paths)):
                # Yolu değiştirmeden doğrudan tam yoldan oku
                char_img = cv.imread(char_path, 0)
                if char_img is None:
                    continue
                ready_char = prepare_char(char_img)
                feature_vector = featurizer(ready_char)
                X.append(feature_vector)  # Girdi verisine ekleme
                Y.append(char)  # Etiket verisine ekleme

    return X, Y  # Girdi ve etiket verilerini döndürme


def train(selected=None):
    global scores, results
    skip = [0, 0, 0, 0]  # Tüm modelleri varsayılan olarak eğit

    # Secim parametre olarak da verilebilir; modul-global durum Streamlit
    # oturumlari arasinda sizdigi icin parametre tercih edilmelidir.
    global selected_models
    if selected is not None:
        selected_models = list(selected)
    # Açık seçim yoksa (parametre None ve global boş) varsayılan davranış:
    # tüm modeller eğitilir. Boş seçim hepsini atlayıp boş rapor yazmasın.
    if not selected_models:
        selected_models = list(ALL_MODELS)
    if "LinearSVM" not in selected_models:
        skip[0] = 1
    if "1L_NN" not in selected_models:
        skip[1] = 1
    if "2L_NN" not in selected_models:
        skip[2] = 1
    if "Gaussian_Naive_Bayes" not in selected_models:
        skip[3] = 1

    scores = []  # Doğruluk skorlarını tutmak için liste oluştur
    # Eğitim işlemi
    X, Y = read_data()
    assert (len(X) == len(Y))

    X, Y = shuffle(X, Y)

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.8)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    scores = []
    results = []
    destination = os.path.join(_MODULE_DIR, 'models')
    os.makedirs(destination, exist_ok=True)

    for idx, clf in tqdm(enumerate(classifiers), desc='Classifiers'):
        if not skip[idx]:  # Belirli modelleri atla
            clf.fit(X_train, Y_train)  # Modeli eğitme
            score = clf.score(X_test, Y_test)  # Test verisi üzerinde doğruluk skoru hesaplama
            scores.append(score)  # Skoru listeye ekle
            # Skor, adiyla birlikte saklanir: names ile zip'lemek model
            # secimi yapildiginda skorlari yanlis modele kaydiriyordu.
            results.append((names[idx], score))
            print(names[idx], score)

            location = os.path.join(destination, f'{names[idx]}.sav')
            pickle.dump(clf, open(location, 'wb'))

    with open(os.path.join(destination, 'report.txt'), 'w') as fo:
        for name, score in results:
            fo.write(f'Score of {name}: {score}\n')


if __name__ == "__main__":
    train()