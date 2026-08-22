import fitz  # PyMuPDF (Sangat akurat untuk PDF UU)
import re
import json

def clean_and_extract_kuhp(pdf_path, output_json_path):
    print("[1/4] Membaca dan mengekstrak teks dari PDF KUHP...")
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        full_text += text + "\n"
        
    print("[2/4] Membersihkan noise (header, footer, watermark, dll)...")
    # 1. Menghapus teks watermark/header standar
    full_text = re.sub(r'PRESIDEN\s*REPUBLIK INDONESIA', '', full_text, flags=re.IGNORECASE)
    full_text = re.sub(r'SALINAN', '', full_text)
    
    # 2. Menghapus nomor halaman (misal: "- 115 -")
    full_text = re.sub(r'\n-\s*\d+\s*-\n', '\n', full_text) 
    
    # 3. Menghapus kode SK administratif di pojok dokumen (misal: SK No 161001 A)
    full_text = re.sub(r'SK\s+No\s+\d+[A-Z\s]*', '', full_text)
    
    # 4. Merapikan spasi dan enter yang berantakan dari PDF
    full_text = re.sub(r'\n\s*\n', '\n', full_text)
    
    print("[3/4] Menerapkan Regex untuk Semantic Chunking berbasis Pasal...")
    # Menggunakan Positive Lookahead (?=Pasal \d+) agar potongan pas di awal kata "Pasal"
    # tanpa menghilangkan kata "Pasal"-nya itu sendiri.
    pasal_pattern = r'(?=Pasal\s+\d+)'
    chunks = re.split(pasal_pattern, full_text)
    
    kuhp_dict = {}
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # Ekstraksi Nama Pasal untuk dijadikan Key (misal: "Pasal 1")
        match = re.match(r'(Pasal\s+\d+)', chunk)
        if match:
            pasal_name = match.group(1)
            # Membersihkan sisa baris baru di dalam chunk teks pasal
            clean_chunk = chunk.replace('\n', ' ')
            # Hapus spasi ganda
            clean_chunk = re.sub(r'\s+', ' ', clean_chunk)
            kuhp_dict[pasal_name] = clean_chunk
        else:
            # Bagian sebelum Pasal 1 (Menimbang, Mengingat, Bab I) dimasukkan ke Pendahuluan
            clean_chunk = chunk.replace('\n', ' ')
            clean_chunk = re.sub(r'\s+', ' ', clean_chunk)
            if "Pendahuluan_dan_Bab_Awal" not in kuhp_dict:
                kuhp_dict["Pendahuluan_dan_Bab_Awal"] = clean_chunk
            else:
                kuhp_dict["Pendahuluan_dan_Bab_Awal"] += " " + clean_chunk
                
    print(f"[INFO] Berhasil mengekstrak {len(kuhp_dict) - 1} Pasal dan 1 Pendahuluan.")
    
    print(f"[4/4] Menyimpan ke format JSON yang siap dipakai RAG: {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(kuhp_dict, f, ensure_ascii=False, indent=4)
        
    print("=== EKSEKUSI FASE 1.3 SELESAI ===")

# --- Eksekusi ---
pdf_file = "KUHP BARU UU Nomor 1 Tahun 2023.pdf" 
output_file = "kuhp_bersih.json"

clean_and_extract_kuhp(pdf_file, output_file)