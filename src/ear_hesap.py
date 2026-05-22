import numpy as np

def goz_uzakligi(nokta_a, nokta_b):
    # İki nokta arasındaki mesafeyi hesapla
    return np.linalg.norm(nokta_a - nokta_b)

def ear_hesapla(goz_noktalari):
    # Dikey mesafeler
    a = goz_uzakligi(goz_noktalari[1], goz_noktalari[5])
    b = goz_uzakligi(goz_noktalari[2], goz_noktalari[4])
    # Yatay mesafe
    c = goz_uzakligi(goz_noktalari[0], goz_noktalari[3])
    # EAR formülü
    ear = (a + b) / (2.0 * c)
    return ear



def mar_hesapla(agiz_noktalari):
    dikey = goz_uzakligi(agiz_noktalari[5], agiz_noktalari[15])  # 0 - 17
    yatay = goz_uzakligi(agiz_noktalari[0], agiz_noktalari[10])  # 61 - 291
    mar = dikey / (2.0 * yatay)
    return mar

