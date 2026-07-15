from atlazer.utils.stanza_chunker import chunk_answer
from atlazer.celery_app.tasks.challenge import chunk_answer as chunk_answer_task

LONG_TEXT = """
Efek Unruh dalam gerak melingkar pada dimensi ruang-waktu 2+1 untuk medan skalar tak bermassa menunjukkan diskrepansi suhu efektif yang jauh lebih rendah dibandingkan prediksi percepatan linear ketika celah energi detektor kecil dan durasi interaksi panjang. Fenomena ini relevan bagi simulasi sistem ruang-waktu analog pada kondensat Bose-Einstein dan film tipis superfluida helium, di mana pemahaman akurat tentang suhu efektif sangat penting untuk verifikasi eksperimental.
Model uplift dalam ekosistem e-commerce skala besar seringkali melanggar asumsi SUTVA, yang menyebabkan ketidakakuratan dalam mengestimasi Individual Treatment Effect (ITE). Terdapat dua masalah utama: kanibalisasi tingkat penjual (seller-level cannibalization), di mana insentif hanya mengalihkan pengeluaran antar toko, dan kanibalisasi tingkat insentif (incentive-level cannibalization), di mana konversi organik atau insentif lain yang bersamaan disalahartikan sebagai dampak dari treatment.
"""


def main():
    # result = chunk_answer(
    #     LONG_TEXT,
    #     lang="id",
    #     semantic=True,          # aktifkan topic-aware chunking
    #     download_models=False,   # cukup sekali di awal
    #     min_words=10,
    # )

    # print(result)

    job = chunk_answer_task.apply_async(
        kwargs={
            "metadata": {
                "user_id": "123",
                "challenge_id": "123",
                "content": LONG_TEXT,
            }
        },
        queue="challenge",
    )
    print(job.id)


if __name__ == '__main__':
    main()
