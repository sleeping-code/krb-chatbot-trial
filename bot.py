from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Replicate
from langchain_core.tools import tool
from dotenv import load_dotenv
import os


def parse_input(input_str: str) -> dict:
    """
    Utility kecil untuk parsing string seperti:
    "key1=value1; key2=value2"
    Jadi dict: {"key1": "value1", "key2": "value2"}
    (dipertahankan biar kalau nanti mentor kamu refer ke fungsi ini, tetap ada.)
    """
    parts = [p.strip() for p in input_str.split(";") if p.strip()]
    result = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


@tool
def kebun_raya_faq(question: str) -> str:
    """
    Tool utama untuk jawab pertanyaan seputar Kebun Raya Bogor:
    - Jam buka
    - Harga tiket (gambaran umum)
    - Lokasi & akses
    - Fasilitas & kendaraan
    - Aturan & piknik
    """
    q = question.lower()

    # Jam buka
    if "jam" in q or "buka" in q or "operasional" in q:
        return (
            "Jam operasional Kebun Raya Bogor (bisa berubah sewaktu-waktu, "
            "jadi sebaiknya cek lagi situs resmi sebelum berkunjung):\n\n"
            "- Senin–Jumat: sekitar 08.00 – 16.00 WIB\n"
            "- Sabtu, Minggu & hari libur nasional: sekitar 07.00 – 16.00 WIB\n\n"
            "Untuk info paling terbaru, cek laman resmi kebunraya.id atau tiketkebunraya.id ya. 😊"
        )

    # Harga tiket
    if "harga" in q or "tiket" in q or "htm" in q:
        return (
            "Harga tiket masuk Kebun Raya Bogor bisa berbeda antara weekday dan weekend "
            "serta dapat berubah sewaktu-waktu.\n\n"
            "Perkiraan (untuk pengunjung domestik):\n"
            "- Weekday: kisaran belasan ribu per orang\n"
            "- Weekend/Libur: kisaran dua puluhan ribu per orang\n\n"
            "Untuk harga resmi & paket lain (kendaraan, membership, dll), "
            "silakan cek langsung di:\n"
            "- Website tiket online: tiketkebunraya.id\n"
            "- Website resmi: kebunraya.id\n"
        )

    # Lokasi / alamat
    if "lokasi" in q or "alamat" in q or "dimana" in q:
        return (
            "Kebun Raya Bogor berlokasi di pusat Kota Bogor, Jawa Barat.\n\n"
            "Alamat kurang lebih:\n"
            "Jl. Ir. H. Juanda, Paledang, Kecamatan Bogor Tengah, Kota Bogor.\n\n"
            "Akses:\n"
            "- Dari Stasiun Bogor bisa jalan kaki / naik transportasi online ke pintu masuk.\n"
            "- Dekat dengan area Surya Kencana & Istana Bogor."
        )

    # Fasilitas & kendaraan
    if (
        "fasilitas" in q
        or "shuttle" in q
        or "golf" in q
        or "sepeda" in q
        or "sewa" in q
        or "kendaraan" in q
    ):
        return (
            "Di Kebun Raya Bogor tersedia beberapa fasilitas yang bisa disewa, misalnya:\n\n"
            "- Shuttle bus untuk keliling area tertentu\n"
            "- Golf car (mobil golf) dengan beberapa kapasitas kursi\n"
            "- Penyewaan sepeda\n"
            "- Kadang tersedia sepeda listrik / e-bike\n\n"
            "Harga sewa bisa berbeda-beda tergantung jenis dan durasi, "
            "jadi untuk angka detail & update terbaru sebaiknya cek langsung ke pihak Kebun Raya "
            "atau lihat di kebunraya.id."
        )

    # Aturan & piknik
    if (
        "aturan" in q
        or "boleh" in q
        or "piknik" in q
        or "bawa makanan" in q
        or "bekal" in q
    ):
        return (
            "Beberapa aturan umum di Kebun Raya Bogor:\n\n"
            "- Boleh piknik & bawa makanan dari luar, tapi gunakan wadah yang bisa dipakai ulang\n"
            "- Jaga kebersihan, buang sampah pada tempatnya\n"
            "- Tidak boleh menancapkan tenda ke tanah\n"
            "- Jangan mengikat spanduk/baliho ke pohon\n"
            "- Acara bermuatan politik tidak diperbolehkan\n"
            "- Tidak diperbolehkan membawa kompor portabel\n\n"
            "Kalau mau mengadakan acara besar, biasanya perlu koordinasi & menyewa lahan resmi."
        )

    # Default jawaban kalau tidak masuk kategori di atas
    return (
        "Aku adalah asisten virtual Kebun Raya Bogor. 🪴\n\n"
        "Aku bisa bantu jawab seputar:\n"
        "- Jam buka & harga tiket\n"
        "- Lokasi & akses\n"
        "- Fasilitas dan aktivitas di Kebun Raya\n"
        "- Aturan umum & info kunjungan\n\n"
        "Kalau pertanyaannya di luar itu, lebih baik kamu cek sumber lain ya, "
        "atau langsung lihat situs resmi kebunraya.id. 🙂"
    )


def build_agent():
    """
    Fungsi yang dipanggil di app.py:
      from bot import build_agent
    dan dipakai sebagai:
      agent = build_agent()
    """
    load_dotenv()

    # REPLICATE_API_TOKEN harus sudah di-set di environment (di Colab pakai os.environ)
    if not os.getenv("REPLICATE_API_TOKEN"):
        # Biar kalau lupa set token, error-nya jelas
        raise ValueError(
            "REPLICATE_API_TOKEN belum diset. "
            "Set dulu, misalnya:\n"
            "import os; os.environ['REPLICATE_API_TOKEN'] = 'token_kamu'"
        )

    llm = Replicate(
        # sesuaikan model dengan yang sudah kamu pakai di kelas kalau perlu
        model="anthropic/claude-3.5-haiku",
    )

    system_message = """
Kamu adalah asisten virtual Kebun Raya Bogor.

Tugasmu:
- Menjawab pertanyaan seputar Kebun Raya Bogor: jam buka, harga tiket, cara beli tiket, fasilitas, aturan, aktivitas, dan info kunjungan lain.
- Gunakan bahasa Indonesia yang sopan, ramah, dan mudah dimengerti. Jika pengguna bertanya dalam bahasa Inggris, kamu boleh menjawab dalam bahasa Inggris.
- Jika menyebut jam buka atau harga, ingatkan bahwa informasi bisa berubah dan sarankan pengguna untuk cek situs resmi kebunraya.id atau tiketkebunraya.id.
- Jika pertanyaan di luar topik Kebun Raya (misalnya politik, gosip, dll), jawab dengan sopan bahwa kamu hanya fokus menjawab hal yang berkaitan dengan Kebun Raya Bogor.
- Jika topik yang ditanyakan cocok dengan tool kebun_raya_faq, gunakan tool tersebut untuk memberikan jawaban yang terstruktur.
"""

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    tools = [kebun_raya_faq]

    agent = initialize_agent(
        llm=llm,
        tools=tools,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs={"system_message": system_message},
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )

    return agent
