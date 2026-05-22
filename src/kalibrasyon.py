import numpy as np


class Kalibrasyon:
    """
    Sistem açılışında sürücüden 5 saniyelik ölçüm alır.
    Pitch, Yaw, Roll ve EAR baseline değerlerini hesaplar.
    Pitch alarm eşiğini kişinin doğal baş hareketi varyansından türetir.
    """

    def __init__(self, sure=150):
        self.sure = sure          # hedef kare sayısı (30 FPS → 5 sn)
        self.sayac = 0
        self.tamamlandi = False

        self._pitch_liste   = []
        self._yaw_liste     = []
        self._roll_liste    = []
        self._sol_ear_liste = []
        self._sag_ear_liste = []

        # Kalibrasyon sonuçları — tamamlandi=True olduktan sonra geçerli
        self.baseline_pitch   = 0.0
        self.baseline_yaw     = 0.0
        self.baseline_roll    = 0.0
        self.baseline_sol_ear = 0.0
        self.baseline_sag_ear = 0.0

        # 3-sigma yöntemiyle kişiye özel pitch eşikleri
        # _hesapla() çağrılmadan önce varsayılan değerler kullanılır
        self.esik_pitch_dusus   = -15.0
        self.esik_pitch_temizle =  -5.0

    def guncelle(self, pitch, yaw, roll, sol_ear, sag_ear):
        """
        Her karede, yüz bulununca çağrılır.
        Kalibrasyon tamamlandığında True döner.
        """
        if self.tamamlandi:
            return True

        self._pitch_liste.append(pitch)
        self._yaw_liste.append(yaw)
        self._roll_liste.append(roll)
        self._sol_ear_liste.append(sol_ear)
        self._sag_ear_liste.append(sag_ear)
        self.sayac += 1

        if self.sayac >= self.sure:
            self._hesapla()
            self.tamamlandi = True

        return self.tamamlandi

    def yuzde(self):
        """İlerleme yüzdesi (0-100)."""
        return min(100, int(self.sayac / self.sure * 100))

    def _hesapla(self):
        self.baseline_pitch   = float(np.mean(self._pitch_liste))
        self.baseline_yaw     = float(np.mean(self._yaw_liste))
        self.baseline_roll    = float(np.mean(self._roll_liste))
        self.baseline_sol_ear = float(np.mean(self._sol_ear_liste))
        self.baseline_sag_ear = float(np.mean(self._sag_ear_liste))

        # Pitch eşiği: 3 × std — min 8°, max 15° (orijinal sabit değerin üzerine çıkmasın)
        # Kalibrasyon gürültüsünün eşiği gereğinden yumuşatmasını önler
        std_pitch = float(np.std(self._pitch_liste))
        self.esik_pitch_dusus   = -float(np.clip(3.0 * std_pitch, 8.0, 15.0))
        self.esik_pitch_temizle =  self.esik_pitch_dusus / 3.0
