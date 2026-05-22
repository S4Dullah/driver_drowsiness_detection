import os
HEADLESS = not os.environ.get('DISPLAY')
if HEADLESS:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2
import numpy as np
import time
import subprocess
import threading
import http.server
import socketserver
from collections import deque
import mediapipe as mp

from gpiozero import LED, Buzzer

from ear_hesap import ear_hesapla, mar_hesapla
from perclos import perclos_hesapla
from headpose import pitch_hesapla
from kalibrasyon import Kalibrasyon

# --- TENSORFLOW LITE ---
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# --- MODEL YOLU (mutlak yol, çalışma dizininden bağımsız) ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.tflite")

# --- KAMERA (picamera2 subprocess — libcamera Python 3.13 için sistem Python kullanır) ---
FRAME_W, FRAME_H = 640, 480
FRAME_BYTES = FRAME_W * FRAME_H * 3
_cam_log = open(os.path.join(BASE_DIR, '..', 'camera_server.log'), 'w')
_cam_proc = subprocess.Popen(
    ['/usr/bin/python3', os.path.join(BASE_DIR, 'camera_server.py')],
    stdout=subprocess.PIPE,
    stderr=_cam_log,
)

class _PicameraCapture:
    def read(self):
        data = _cam_proc.stdout.read(FRAME_BYTES)
        if len(data) < FRAME_BYTES:
            return False, None
        frame = np.frombuffer(data, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 3)
        return True, frame.copy()

    def isOpened(self):
        return _cam_proc.poll() is None

    def release(self):
        _cam_proc.terminate()

cap = _PicameraCapture()
time.sleep(1)  # picamera2 başlangıç süresi

# --- GPIO LEDs + BUZZER ---
# Kırmızı: GPIO17 Pin11/Pin9  | Yeşil: GPIO27 Pin13/Pin20 | Sarı: GPIO22 Pin15/Pin25
# Buzzer : GPIO18 Pin12/Pin14
led_kirmizi = LED(17)
led_yesil   = LED(27)
led_sari    = LED(22)
buzzer      = Buzzer(18)

# --- MJPEG HTTP STREAM (port 8080) ---
_mjpeg_frame = None
_mjpeg_lock  = threading.Lock()

class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                with _mjpeg_lock:
                    frame = _mjpeg_frame
                if frame is not None:
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    data = buf.tobytes()
                    self.wfile.write(
                        b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
                    )
                time.sleep(0.05)
        except Exception:
            pass

def _mjpeg_sunucu_baslat(port=8080):
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), _MJPEGHandler) as httpd:
        httpd.serve_forever()

threading.Thread(target=_mjpeg_sunucu_baslat, daemon=True).start()

mevcut_led_durumu    = None
mevcut_buzzer_durumu = None

def _led_kapat():
    led_kirmizi.off()
    led_yesil.off()
    led_sari.off()

def led_guncelle(yeni_durum):
    """3 LED state machine — değişmiyorsa dokunmaz."""
    global mevcut_led_durumu
    if yeni_durum == mevcut_led_durumu:
        return
    _led_kapat()
    if yeni_durum == "kalibrasyon":
        led_yesil.blink(on_time=0.3, off_time=0.3, background=True)
        led_sari.blink(on_time=0.3, off_time=0.3, background=True)
    elif yeni_durum == "yuz_yok":
        led_sari.on()
    elif yeni_durum == "normal":
        led_yesil.on()
    elif yeni_durum == "ear_gate":
        led_sari.on()
        led_yesil.on()
    elif yeni_durum == "esneme":
        led_sari.on()
    elif yeni_durum == "uyku":
        led_kirmizi.on()
    elif yeni_durum == "kritik":
        led_kirmizi.blink(on_time=0.15, off_time=0.15, background=True)
    # "off" → _led_kapat() ile zaten söndürüldü
    mevcut_led_durumu = yeni_durum

def buzzer_guncelle(yeni_durum):
    """Buzzer durumunu state machine ile yönetir — değişmiyorsa dokunmaz."""
    global mevcut_buzzer_durumu
    if yeni_durum == mevcut_buzzer_durumu:
        return
    buzzer.off()
    if yeni_durum == "on":
        # Sürekli ses — kafa düşme gibi kritik durum
        buzzer.beep(on_time=0.1, off_time=0.1, background=True)
    elif yeni_durum == "uyari":
        # Yavaş bip — uyku tespiti
        buzzer.beep(on_time=0.5, off_time=0.5, background=True)
    # "off" durumu zaten buzzer.off() ile yapıldı
    mevcut_buzzer_durumu = yeni_durum

# --- MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# --- TFLite Model ---
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- PARAMETRE AYARLARI ---
ESIK_AI_DEGER     = 0.80
ESIK_PERCLOS      = 0.20
ESIK_MAR          = 0.45
EAR_DUSUS_YUZDESI = 0.65
ESIK_YAW          = 35.0
ESIK_ROLL         = 25.0

# Her ANALIZ_ATLAMA karede bir MediaPipe + AI çalışır (~10 FPS analiz, ~30 FPS görüntü)
ANALIZ_ATLAMA = 3

# 150 analiz karesi × (ANALIZ_ATLAMA/30 FPS) ≈ 15 sn kalibrasyon süresi
kal = Kalibrasyon(sure=150)

kafa_dustu_alarm_aktif = False
kafa_dusus_sayac       = 0
KAFA_DUSUS_ONAY        = 5   # art arda kaç analiz karesi gerekli (~0.5 sn)

# --- BELLEK YAPILARI ---
# PERCLOS: 100 analiz karesi × 0.1 sn = 10 sn pencere
ear_durumu_gecmisi  = deque(maxlen=100)
ai_durumu_gecmisi   = deque(maxlen=100)
agız_durumu_gecmisi = deque(maxlen=100)

# AI zaman yumuşatma: 3 analiz karesi ≈ 0.3 sn
ZAMAN_PENCERESI_SOL = deque(maxlen=3)
ZAMAN_PENCERESI_SAG = deque(maxlen=3)

# Pitch yumuşatma: 3 analiz karesi ≈ 0.3 sn
pitch_gecmisi = deque(maxlen=3)

# EAR gate yumuşatma: 5 analiz karesi ≈ 0.5 sn
yaw_gecmisi  = deque(maxlen=5)
roll_gecmisi = deque(maxlen=5)

# EAR anti-parlama yumuşatma: 2 analiz karesi ≈ 0.2 sn
sol_ear_gecmisi = deque(maxlen=2)
sag_ear_gecmisi = deque(maxlen=2)
son_gecerli_sol_ear = 0.30
son_gecerli_sag_ear = 0.30

SON_YUZ_ZAMANI = time.time()

SOL_GOZ_IDX = [33, 160, 158, 133, 153, 144]
SAG_GOZ_IDX = [362, 385, 387, 263, 373, 380]
AGIZ_IDX    = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


def get_coords(face_landmarks, indices, w, h):
    return np.array(
        [[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in indices],
        dtype=np.float64
    )


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def goz_kirp_ve_tahmin_tflite(kare, goz_noktalari, zaman_penceresi, parlama_esigi, w, h):
    pay = 15
    x_min = int(np.min(goz_noktalari[:, 0])) - pay
    x_max = int(np.max(goz_noktalari[:, 0])) + pay
    y_min = int(np.min(goz_noktalari[:, 1])) - pay
    y_max = int(np.max(goz_noktalari[:, 1])) + pay

    goz_genislik  = x_max - x_min
    goz_yukseklik = y_max - y_min
    kare_kenar    = max(goz_genislik, goz_yukseklik)
    merkez_x      = x_min + goz_genislik // 2
    merkez_y      = y_min + goz_yukseklik // 2

    yeni_x_min = max(0, merkez_x - kare_kenar // 2)
    yeni_x_max = min(w,  merkez_x + kare_kenar // 2)
    yeni_y_min = max(0, merkez_y - kare_kenar // 2)
    yeni_y_max = min(h,  merkez_y + kare_kenar // 2)

    kirpilmis = kare[yeni_y_min:yeni_y_max, yeni_x_min:yeni_x_max]
    if kirpilmis.size == 0:
        return 0, 0.0, False

    hsv_goz = cv2.cvtColor(kirpilmis, cv2.COLOR_BGR2HSV)
    _, parlama_maskesi = cv2.threshold(hsv_goz[:, :, 2], parlama_esigi, 255, cv2.THRESH_BINARY)
    parlama_orani = cv2.countNonZero(parlama_maskesi) / (kirpilmis.shape[0] * kirpilmis.shape[1] + 1e-6)
    parlama_aktif = parlama_orani > 0.005

    kirpilmis_rgb = cv2.cvtColor(kirpilmis, cv2.COLOR_BGR2RGB)
    girdi = np.expand_dims(cv2.resize(kirpilmis_rgb, (128, 128)) / 255.0, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], girdi)
    interpreter.invoke()
    tahmin = interpreter.get_tensor(output_details[0]['index'])[0]
    ham_uyanik, ham_uyku = tahmin[0], tahmin[1]

    if parlama_aktif:
        if len(zaman_penceresi) > 0:
            ham_uyanik, ham_uyku = zaman_penceresi[-1]
        else:
            ham_uyanik, ham_uyku = 0.51, 0.49

    zaman_penceresi.append((ham_uyanik, ham_uyku))
    ortalama_uyku = np.mean([t[1] for t in zaman_penceresi])
    return (1 if ortalama_uyku > 0.50 else 0), float(ortalama_uyku), parlama_aktif


def main():
    global kafa_dustu_alarm_aktif, kafa_dusus_sayac, SON_YUZ_ZAMANI
    global son_gecerli_sol_ear, son_gecerli_sag_ear

    fps_hesap_zaman  = time.time()
    kare_sayisi      = 0
    son_debug_zamani = time.time()
    debug_metinleri  = []
    delta_pitch      = 0.0

    # Atlanan kareler için önbellek — analiz sonuçları buradan okunur
    son_durum_metni        = "Baslaniyor..."
    son_durum_rengi        = (128, 128, 128)
    son_landmark_noktalari = []
    son_kal_bilgi          = ""

    # NOT: cv2.imshow Pi'ye HDMI ekran bağlıysa çalışır.
    # Headless kullanımda bu satırı ve cv2.imshow çağrısını kaldır.

    try:
        while True:
            ret, kare = cap.read()
            if not ret:
                print("Kamera karesi okunamadı — bağlantıyı kontrol edin.")
                break

            h, w, _ = kare.shape
            kare_sayisi += 1

            # Parlaklik ölçümü ve gamma düzeltme her karede çalışır (LUT: çok ucuz)
            gri_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
            ortalama_parlaklik    = np.mean(gri_kare)
            dinamik_parlama_esigi = 180
            if ortalama_parlaklik < 60:
                dinamik_parlama_esigi = 235
                kare = adjust_gamma(kare, gamma=2.2)

            # =================================================================
            # AĞIR ANALİZ — her ANALIZ_ATLAMA karede bir (~10 FPS)
            # =================================================================
            if kare_sayisi % ANALIZ_ATLAMA == 0:
                rgb_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
                results  = face_mesh.process(rgb_kare)

                if results.multi_face_landmarks:
                    SON_YUZ_ZAMANI = time.time()
                    face_landmarks = results.multi_face_landmarks[0]

                    sol_goz  = get_coords(face_landmarks, SOL_GOZ_IDX, w, h)
                    sag_goz  = get_coords(face_landmarks, SAG_GOZ_IDX, w, h)
                    agiz_nok = get_coords(face_landmarks, AGIZ_IDX, w, h)

                    anlik_pitch, anlik_yaw, anlik_roll = pitch_hesapla(face_landmarks, w, h)
                    yaw_gecmisi.append(anlik_yaw)
                    roll_gecmisi.append(anlik_roll)
                    yumusak_yaw  = float(np.mean(yaw_gecmisi))
                    yumusak_roll = float(np.mean(roll_gecmisi))
                    sol_ear, sag_ear = ear_hesapla(sol_goz), ear_hesapla(sag_goz)

                    # Landmark noktalarını atlanan kareler için önbelleğe al
                    son_landmark_noktalari = [
                        (int(face_landmarks.landmark[idx].x * w),
                         int(face_landmarks.landmark[idx].y * h))
                        for idx in SOL_GOZ_IDX + SAG_GOZ_IDX
                    ]

                    # --- 1. KALİBRASYON ---
                    if not kal.tamamlandi:
                        kal.guncelle(anlik_pitch, anlik_yaw, anlik_roll, sol_ear, sag_ear)
                        son_kal_bilgi = (
                            f"EAR: {sol_ear:.3f}/{sag_ear:.3f}  "
                            f"YAW: {anlik_yaw:.1f}  ROLL: {anlik_roll:.1f}"
                        )
                        led_guncelle("kalibrasyon")

                    # --- 2. PITCH TAKİBİ (kalibrasyon sonrası) ---
                    else:
                        pitch_gecmisi.append(anlik_pitch)
                        delta_pitch = np.mean(pitch_gecmisi) - kal.baseline_pitch

                        if delta_pitch < kal.esik_pitch_dusus:
                            kafa_dusus_sayac += 1
                            if kafa_dusus_sayac >= KAFA_DUSUS_ONAY:
                                kafa_dustu_alarm_aktif = True
                        elif delta_pitch > kal.esik_pitch_temizle:
                            kafa_dusus_sayac       = 0
                            kafa_dustu_alarm_aktif = False

                    # --- 3. GÖZ ANALİZİ (kalibrasyon sonrası) ---
                    if kal.tamamlandi:
                        agiz_mar = mar_hesapla(agiz_nok)

                        sol_ai, sol_guven, sol_parlama = goz_kirp_ve_tahmin_tflite(
                            kare, sol_goz, ZAMAN_PENCERESI_SOL, dinamik_parlama_esigi, w, h)
                        sag_ai, sag_guven, sag_parlama = goz_kirp_ve_tahmin_tflite(
                            kare, sag_goz, ZAMAN_PENCERESI_SAG, dinamik_parlama_esigi, w, h)

                        if sol_parlama:
                            sol_ear_gecmisi.append(son_gecerli_sol_ear)
                            kullanilacak_sol_ear = son_gecerli_sol_ear
                        else:
                            sol_ear_gecmisi.append(sol_ear)
                            kullanilacak_sol_ear = np.mean(sol_ear_gecmisi)
                            son_gecerli_sol_ear  = kullanilacak_sol_ear

                        if sag_parlama:
                            sag_ear_gecmisi.append(son_gecerli_sag_ear)
                            kullanilacak_sag_ear = son_gecerli_sag_ear
                        else:
                            sag_ear_gecmisi.append(sag_ear)
                            kullanilacak_sag_ear = np.mean(sag_ear_gecmisi)
                            son_gecerli_sag_ear  = kullanilacak_sag_ear

                        esik_sol_ear = kal.baseline_sol_ear * EAR_DUSUS_YUZDESI
                        esik_sag_ear = kal.baseline_sag_ear * EAR_DUSUS_YUZDESI

                        fiziksel_sol_kapali = kullanilacak_sol_ear < esik_sol_ear
                        fiziksel_sag_kapali = kullanilacak_sag_ear < esik_sag_ear

                        ear_guvensiz = (abs(yumusak_yaw  - kal.baseline_yaw)  > ESIK_YAW or
                                        abs(yumusak_roll - kal.baseline_roll) > ESIK_ROLL)

                        fiziksel_kapali = (fiziksel_sol_kapali and fiziksel_sag_kapali) and not ear_guvensiz
                        ai_kapali       = (sol_guven > ESIK_AI_DEGER) and (sag_guven > ESIK_AI_DEGER)

                        ear_durumu_gecmisi.append(1 if fiziksel_kapali else 0)
                        ai_durumu_gecmisi.append(1 if ai_kapali else 0)
                        perclos_ear = perclos_hesapla(ear_durumu_gecmisi)
                        perclos_ai  = perclos_hesapla(ai_durumu_gecmisi)

                        agız_acık = 1 if agiz_mar > ESIK_MAR else 0
                        agız_durumu_gecmisi.append(agız_acık)
                        permar = perclos_hesapla(agız_durumu_gecmisi)

                        aktif_perclos_esigi = 0.18 if permar > 0.30 else ESIK_PERCLOS

                        # --- KARAR HİYERARŞİSİ + LED ---
                        if kafa_dustu_alarm_aktif:
                            son_durum_metni = "KRITIK: KAFA DUSTU! UYAN!"
                            son_durum_rengi = (0, 0, 255)
                            led_guncelle("kritik")
                            buzzer_guncelle("on")

                        elif perclos_ear >= aktif_perclos_esigi or perclos_ai >= aktif_perclos_esigi:
                            son_durum_metni = "TEHLIKE: UYKU TESPIT EDILDI!"
                            son_durum_rengi = (0, 0, 255)
                            led_guncelle("uyku")
                            buzzer_guncelle("uyari")

                        elif permar > 0.30:
                            son_durum_metni = "BILGI: Yorgunluk Belirtisi (Esneme)"
                            son_durum_rengi = (0, 165, 255)
                            led_guncelle("esneme")
                            buzzer_guncelle("off")

                        elif ear_guvensiz:
                            son_durum_metni = "DIKKAT: EAR GUVENSIZ (Bas donuk)"
                            son_durum_rengi = (0, 255, 255)
                            led_guncelle("ear_gate")
                            buzzer_guncelle("off")

                        else:
                            son_durum_metni = "Surucu Durumu: NORMAL"
                            son_durum_rengi = (0, 255, 0)
                            led_guncelle("normal")
                            buzzer_guncelle("off")

                        if time.time() - son_debug_zamani > 0.3:
                            pitch_durum = "ALARM" if kafa_dustu_alarm_aktif else (
                                "YAKIN" if delta_pitch < kal.esik_pitch_dusus * 0.7 else "OK")
                            debug_metinleri = [
                                f"SOL EAR: {kullanilacak_sol_ear:.3f} / Esik: {esik_sol_ear:.3f}",
                                f"SAG EAR: {kullanilacak_sag_ear:.3f} / Esik: {esik_sag_ear:.3f}",
                                f"PITCH d:{delta_pitch:.1f} / Esik:{kal.esik_pitch_dusus:.1f} [{pitch_durum}]",
                                f"YAW: {yumusak_yaw:.1f}(d:{yumusak_yaw - kal.baseline_yaw:+.1f})  "
                                f"ROLL: {yumusak_roll:.1f}(d:{yumusak_roll - kal.baseline_roll:+.1f})"
                                f"{'  [EAR GATE]' if ear_guvensiz else ''}",
                                f"SOL AI: %{sol_guven * 100:.1f}  |  SAG AI: %{sag_guven * 100:.1f}",
                                f"PERCLOS EAR:%{perclos_ear * 100:.1f} AI:%{perclos_ai * 100:.1f} "
                                f"Esik:%{aktif_perclos_esigi * 100:.1f}",
                            ]
                            son_debug_zamani = time.time()

                else:
                    son_landmark_noktalari = []
                    if time.time() - SON_YUZ_ZAMANI > 2.0:
                        son_durum_metni = "SISTEM KOR: YUZ BULUNAMADI"
                        son_durum_rengi = (0, 0, 255)
                        led_guncelle("yuz_yok")
                        buzzer_guncelle("off")

            # =================================================================
            # GÖRÜNTÜ KATMANI — her karede önbellekten çizilir (~30 FPS)
            # =================================================================

            # Kalibrasyon ilerlemesi
            if not kal.tamamlandi:
                yuzde = kal.yuzde()
                cv2.rectangle(kare, (0, 0), (w, 110), (0, 0, 0), -1)
                cv2.rectangle(kare, (10, 70), (10 + int((w - 20) * yuzde / 100), 90), (0, 255, 255), -1)
                cv2.putText(kare, f"KALIBRASYON %{yuzde}  —  Duz bakin, hareketsiz kalin",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                if son_kal_bilgi:
                    cv2.putText(kare, son_kal_bilgi,
                                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # Sürücü durum metni ve debug paneli
            if kal.tamamlandi:
                kalinlik = 3 if "KRITIK" in son_durum_metni else 2
                cv2.putText(kare, son_durum_metni, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, son_durum_rengi, kalinlik)

                for i, metin in enumerate(debug_metinleri):
                    if "ALARM" in metin or "GATE" in metin:
                        renk = (0, 0, 255)
                    elif "YAKIN" in metin:
                        renk = (0, 165, 255)
                    elif "PERCLOS" in metin:
                        renk = (0, 165, 255)
                    elif "AI" in metin:
                        renk = (0, 255, 255)
                    else:
                        renk = (255, 255, 255)
                    cv2.putText(kare, metin, (10, 80 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1)

            # Landmark noktaları
            for x_l, y_l in son_landmark_noktalari:
                cv2.circle(kare, (x_l, y_l), 1, (0, 255, 0), -1)

            # FPS sayacı (tüm görüntü karelerini ölçer)
            gecen_zaman = time.time() - fps_hesap_zaman
            if gecen_zaman > 1.0:
                fps             = kare_sayisi / gecen_zaman
                kare_sayisi     = 0
                fps_hesap_zaman = time.time()
                cv2.putText(kare, f"FPS: {int(fps)}", (w - 100, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            with _mjpeg_lock:
                _mjpeg_frame = kare.copy()

            if not HEADLESS:
                cv2.imshow("Gomulu Surucu Yorgunluk Sistemi - Pi4", kare)
                if cv2.waitKey(1) == ord('q'):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        _led_kapat()
        led_kirmizi.close()
        led_yesil.close()
        led_sari.close()
        buzzer.off()
        buzzer.close()
        cap.release()
        _cam_log.close()
        if not HEADLESS:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
