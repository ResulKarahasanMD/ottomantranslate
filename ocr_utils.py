# ocr_utils.py

import os
import cv2 as cv
from character_segmentation import segment
from train import prepare_char, featurizer, chars as VALID_CHARS
import pickle

model_name = '2L_NN.sav'

# Geçerli tahmin edilebilecek sınıflar (çok karakterli 'لا' ve noktalama '،' dahil)
_VALID_CHAR_SET = set(VALID_CHARS)

# Model her kelimede yeniden yüklenmesin diye süreç-global olarak bir kez tutulur.
_MODEL = None


def load_model():
    location = 'models'
    if os.path.exists(location):
        model = pickle.load(open(f'models/{model_name}', 'rb'))
        return model
    return None


def init_worker():
    """multiprocessing.Pool başlatıcısı: her worker modeli bir kez yükler."""
    global _MODEL
    _MODEL = load_model()


def _get_model():
    """Modeli süreç-global cache üzerinden verir; yoksa yükler (tekil süreç kullanımı için)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def run2(obj):
    word, line = obj
    model = _get_model()
    if model is None:
        return ''

    # Segmentasyon tek bir kelimede patlarsa tüm sayfayı çökertmesin.
    try:
        char_imgs = segment(line, word)
    except Exception as e:
        print(f"Segmentasyon hatası (kelime atlandı): {e}")
        return ''

    txt_word = ''
    for char_img in char_imgs:
        try:
            ready_char = prepare_char(char_img)
        except Exception as e:
            print(f"Karakter hazırlama hatası: {e}")
            continue
        try:
            feature_vector = featurizer(ready_char)
            predicted_char = model.predict([feature_vector])[0]
        except Exception as e:
            print(f"Tahmin hatası: {e}")
            continue

        # Modelin eğitildiği geçerli sınıflardan biriyse ekle
        # ('لا' gibi çok karakterli ve '،' gibi noktalama sınıfları dahil).
        if predicted_char in _VALID_CHAR_SET:
            txt_word += predicted_char
        else:
            print(f"Geçersiz karakter tahmini: {predicted_char}")
    return txt_word
