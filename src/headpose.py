import cv2
import numpy as np

BAS_POZU_IDX = [1, 9, 57, 130, 287, 359, 33, 263, 61, 291, 199, 175]
MODEL_3D = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -33.0, -6.5],
    [-22.5, -17.0, -13.5],
    [-15.0, 17.0, -12.5],
    [22.5, -17.0, -13.5],
    [15.0, 17.0, -12.5],
    [-30.0, 20.0, -15.0],
    [30.0, 20.0, -15.0],
    [-20.0, -10.0, -15.0],
    [20.0, -10.0, -15.0],
    [0.0, -60.0, -10.0],
    [0.0, -50.0, -8.0],
], dtype=np.float64)

# Kameranın fiziksel montaj açısına göre derece cinsinden düzeltme.
# RPi kamerası farklı bir açıda monte edildiyse burası ayarlanmalı.
MONTAJ_ACISI_DUZELTME = 0.0

# Gerçek kamera kalibrasyonu olmadan sıfır distorsiyon varsayılıyor.
DIST_COEFFS = np.zeros((4, 1))


def pitch_hesapla(face_landmarks, w, h):
    """
    Baş pozunu hesaplar.
    Döndürür: (pitch, yaw, roll) — derece cinsinden.
      pitch : negatif = baş öne düşme  (R[2,1] bileşeni)
      yaw   : pozitif = sağa dönüş    (R[2,0] bileşeni)
      roll  : pozitif = sağa yatma    (R[0,1] bileşeni)

    Neden bu bileşenler?
      Kamera önünde düz duran yüz için R0 ≈ diag(1,-1,-1).
      R[2,1]: sadece öne eğilmeyle (Rx) değişir, yaw/roll'dan bağımsız.
      R[2,0]: sadece yaw (Ry) ile değişir, pitch/roll'dan bağımsız.
      R[0,1]: sadece roll (Rz) ile değişir, pitch/yaw'dan bağımsız.
    """
    goruntu_noktalari = np.array([
        [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
        for i in BAS_POZU_IDX
    ], dtype=np.float64)

    odak = w
    kamera_matrisi = np.array([[odak, 0, w / 2], [0, odak, h / 2], [0, 0, 1]], dtype=np.float64)
    success, rot_vec, _ = cv2.solvePnP(MODEL_3D, goruntu_noktalari, kamera_matrisi, DIST_COEFFS)

    if not success:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rot_vec)

    pitch = np.degrees(np.arcsin(np.clip( rot_mat[2, 1], -1.0, 1.0)))  # öne eğilme — negatif = baş düşüyor
    yaw   = np.degrees(np.arcsin(np.clip(-rot_mat[2, 0], -1.0, 1.0)))  # sağa/sola dönüş
    roll  = np.degrees(np.arcsin(np.clip( rot_mat[0, 1], -1.0, 1.0)))  # omuza yatma

    return float(pitch + MONTAJ_ACISI_DUZELTME), float(yaw), float(roll)
