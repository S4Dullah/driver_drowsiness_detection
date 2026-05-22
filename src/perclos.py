
def perclos_hesapla(goz_durum_gecmisi):
    if len(goz_durum_gecmisi) == 0:
        return 0.0
    toplam_kapalı_kare=sum(goz_durum_gecmisi)
    toplam_kare = len(goz_durum_gecmisi)

    perclos=toplam_kapalı_kare/toplam_kare
    return perclos