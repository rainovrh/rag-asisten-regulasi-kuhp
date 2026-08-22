import pandas as pd
import warnings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

def main():
    print("="*70)
    print("FASE 5: EVALUASI SAINTIFIK MENGGUNAKAN RAGAS (LOCAL LLAMA-3)")
    print("="*70)

    input_file = "hasil_generasi_llama3.csv"
    print(f"[1/4] Membaca data hasil generasi dari '{input_file}'...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: File {input_file} tidak ditemukan!")
        return

    # RAGAS membutuhkan format kolom yang spesifik: question, answer, contexts
    # Contexts harus berupa list of strings
    print("[2/4] Memformat dataset agar kompatibel dengan kerangka RAGAS...")
    
    questions = df['Query'].astype(str).tolist()
    answers = df['AI_Answer'].astype(str).tolist()
    
    # Memastikan konteks diubah menjadi list of string
    contexts = []
    for ctx in df['Retrieved_Context']:
        if pd.isna(ctx) or not str(ctx).strip():
            contexts.append([""]) # Jika kosong karena dibuang CRAG
        else:
            contexts.append([str(ctx)])

    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
    
    ragas_dataset = Dataset.from_dict(data_dict)

    print("[3/4] Menyiapkan LLM (Ollama) sebagai Juri Evaluator...")
    # Kita menggunakan Llama-3 yang sama untuk menilai kualitas jawabannya sendiri
    evaluator_llm = ChatOllama(model="llama3", temperature=0)
    
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    evaluator_embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})

    print("[4/4] 🚀 Memulai Evaluasi RAGAS (Faithfulness & Answer Relevancy)...")
    print("      (Ini akan memakan waktu karena LLM harus membaca dan menilai 50 baris data)")
    
    # Menjalankan fungsi evaluasi RAGAS
    try:
        result = evaluate(
            dataset = ragas_dataset,
            metrics = [faithfulness, answer_relevancy],
            llm = evaluator_llm,
            embeddings = evaluator_embeddings,
        )
    except Exception as e:
        print(f"\n❌ Terjadi kesalahan saat evaluasi RAGAS: {e}")
        return

    print("\n" + "="*70)
    print("🏆 HASIL EVALUASI KESELURUHAN (RATA-RATA):")
    print(result)
    print("="*70)

    # Menyimpan hasil skor per baris kembali ke CSV
    print("\n💾 Menyimpan rincian skor ke file CSV baru...")
    result_df = result.to_pandas()
    
    # Gabungkan dengan kolom asli agar mudah dibaca di Excel
    final_df = pd.concat([df, result_df[['faithfulness', 'answer_relevancy']]], axis=1)
    
    output_file = "hasil_akhir_ragas_skripsi.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"✅ Selesai! File '{output_file}' berhasil dibuat.")
    print("File ini mengandung 'Data Angka' yang Anda butuhkan untuk menyusun Bab 4!")

if __name__ == "__main__":
    main()