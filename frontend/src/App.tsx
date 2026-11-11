import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './index.css'

/* ─── Types ─── */
interface Citation {
  pasal: string
  skor: number
  teks: string
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  isRefusal?: boolean
}

/* ─── Constants ─── */
const API_URL = 'http://localhost:8000/api/chat'

const SUGGESTIONS = [
  {
    icon: '⚖️',
    title: 'Hukuman Korupsi',
    text: 'Apa hukuman untuk tindak pidana korupsi menurut KUHP Baru?',
  },
  {
    icon: '🏛️',
    title: 'Pasal Makar',
    text: 'Jelaskan tentang pasal Makar dalam KUHP Baru',
  },
  {
    icon: '📜',
    title: 'Penghinaan Presiden',
    text: 'Bagaimana aturan tentang penghinaan presiden?',
  },
  {
    icon: '🔍',
    title: 'Pencurian',
    text: 'Apa sanksi pidana untuk pencurian menurut KUHP Baru?',
  },
]

/* ─── App ─── */
export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const sendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text.trim(),
    }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    try {
      const { data } = await axios.post(API_URL, { query: text.trim() })
      const aiMsg: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.answer,
        citations: data.citations,
        isRefusal: data.is_refusal,
      }
      setMessages(prev => [...prev, aiMsg])
    } catch {
      setMessages(prev => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content:
            '⚠️ Gagal terhubung ke server. Pastikan backend FastAPI sedang berjalan di port 8000.',
          isRefusal: true,
        },
      ])
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage(input)
  }

  const hasMessages = messages.length > 0

  /* ─── Input bar component (reused in both states) ─── */
  const InputBar = ({ className = '' }: { className?: string }) => (
    <form onSubmit={handleSubmit} className={`w-full max-w-2xl mx-auto ${className}`}>
      <div className="flex items-center gap-3 bg-white border border-gray-200 rounded-2xl px-4 py-2
                      shadow-[0_2px_20px_rgba(0,0,0,0.06)]
                      focus-within:border-navy-400 focus-within:shadow-[0_2px_24px_rgba(30,58,138,0.12)]
                      transition-all duration-300">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Tanyakan tentang pasal KUHP Baru…"
          disabled={isLoading}
          className="flex-1 bg-transparent outline-none text-sm text-gray-800 py-2
                     placeholder:text-gray-400 disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="w-9 h-9 rounded-xl bg-navy-600 text-white flex items-center justify-center shrink-0
                     hover:bg-navy-700 active:scale-[0.92]
                     disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-navy-600
                     transition-all duration-150 cursor-pointer"
        >
          {isLoading ? (
            <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
              <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
              <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
            </svg>
          )}
        </button>
      </div>
      <p className="text-center text-[10px] text-gray-400 mt-2.5">
        Lex RAG dapat membuat kesalahan · Hasil AI bukan nasihat hukum resmi
      </p>
    </form>
  )

  /* ══════════════════════════════════════════════
     EMPTY STATE — everything centered on screen
     ══════════════════════════════════════════════ */
  if (!hasMessages) {
    return (
      <div className="h-screen flex flex-col items-center justify-center bg-surface px-4">
        {/* ── Logo ── */}
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-navy-500 to-navy-800
                        flex items-center justify-center shadow-lg shadow-navy-600/20 mb-6">
          <span className="text-3xl leading-none">⚖️</span>
        </div>

        {/* ── Title ── */}
        <h1 className="text-4xl font-extrabold text-gray-900 tracking-tight mb-2 text-center">
          Lex RAG
        </h1>
        <p className="text-gray-500 text-sm max-w-sm text-center leading-relaxed mb-10">
          Asisten cerdas untuk menelusuri dan memahami pasal-pasal
          KUHP Baru (UU No. 1 Tahun 2023)
        </p>

        {/* ── Suggestion cards ── */}
        <div className="grid grid-cols-2 gap-3 w-full max-w-xl mb-8">
          {SUGGESTIONS.map(s => (
            <button
              key={s.title}
              onClick={() => sendMessage(s.text)}
              className="group flex flex-col gap-2 p-5 rounded-2xl border border-gray-200 bg-white
                         hover:border-navy-300 hover:shadow-lg hover:shadow-navy-600/5 hover:-translate-y-0.5
                         active:scale-[0.98]
                         transition-all duration-200 text-left cursor-pointer"
            >
              <span className="text-2xl">{s.icon}</span>
              <div>
                <p className="text-sm font-semibold text-gray-800 group-hover:text-navy-700 transition-colors">
                  {s.title}
                </p>
                <p className="text-xs text-gray-400 mt-1 leading-relaxed line-clamp-2">{s.text}</p>
              </div>
            </button>
          ))}
        </div>

        {/* ── Input bar ── */}
        <InputBar className="px-2" />
      </div>
    )
  }

  /* ══════════════════════════════════════════════
     CHAT STATE — messages + sticky input at bottom
     ══════════════════════════════════════════════ */
  return (
    <div className="h-screen flex flex-col bg-surface">
      {/* ── Top bar ── */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-gray-100 bg-white/80 backdrop-blur-md shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-navy-600 flex items-center justify-center">
            <span className="text-white text-sm leading-none">⚖️</span>
          </div>
          <span className="text-sm font-bold text-navy-800">Lex RAG</span>
        </div>
        <button
          onClick={() => { setMessages([]); setInput('') }}
          className="text-xs text-gray-400 hover:text-gray-600 bg-gray-100 hover:bg-gray-200
                     px-3 py-1.5 rounded-lg transition-colors cursor-pointer"
        >
          + Percakapan Baru
        </button>
      </header>

      {/* ── Messages ── */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-4 py-6 space-y-6">
          {messages.map((msg, idx) => (
            <div
              key={msg.id}
              className="animate-fade-in-up"
              style={{ animationDelay: `${Math.min(idx * 0.04, 0.2)}s` }}
            >
              {msg.role === 'user' ? (
                /* ── User ── */
                <div className="flex justify-end">
                  <div className="max-w-[80%] bg-navy-600 text-white rounded-2xl rounded-br-sm px-5 py-3.5 shadow-md shadow-navy-600/10">
                    <p className="text-[14px] leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  </div>
                </div>
              ) : (
                /* ── Assistant ── */
                <div className="flex gap-3">
                  <div className="w-7 h-7 rounded-lg bg-navy-50 border border-navy-100 flex items-center justify-center shrink-0 mt-0.5">
                    <span className="text-xs">⚖️</span>
                  </div>
                  <div className="flex-1 space-y-3 min-w-0">
                    {/* Answer */}
                    <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm">
                      <p className="text-[14px] text-gray-800 leading-[1.7] whitespace-pre-wrap">{msg.content}</p>
                    </div>

                    {/* Citations */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="space-y-1.5 pl-1">
                        <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400 mb-2">
                          📑 Sumber Pasal
                        </p>
                        {msg.citations.map((c, i) => (
                          <details
                            key={i}
                            className="group bg-white border border-gray-100 rounded-xl overflow-hidden
                                       hover:border-navy-200 transition-colors"
                          >
                            <summary className="flex items-center justify-between px-4 py-2.5 cursor-pointer select-none text-sm">
                              <span className="font-semibold text-navy-700 text-xs bg-navy-50 px-2 py-0.5 rounded-md">
                                {c.pasal.toUpperCase()}
                              </span>
                              <span className="text-[10px] font-mono text-gray-400">
                                {c.skor.toFixed(3)}
                              </span>
                            </summary>
                            <div className="px-4 pb-3 border-t border-gray-50">
                              <p className="text-xs text-gray-600 leading-relaxed pt-2">{c.teks}</p>
                            </div>
                          </details>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Typing indicator */}
          {isLoading && (
            <div className="flex gap-3 animate-fade-in-up">
              <div className="w-7 h-7 rounded-lg bg-navy-50 border border-navy-100 flex items-center justify-center shrink-0">
                <span className="text-xs">⚖️</span>
              </div>
              <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="flex gap-1">
                    <span className="typing-dot w-2 h-2 rounded-full bg-navy-300" />
                    <span className="typing-dot w-2 h-2 rounded-full bg-navy-300" />
                    <span className="typing-dot w-2 h-2 rounded-full bg-navy-300" />
                  </div>
                  <span className="text-xs text-gray-400">Menganalisis KUHP…</span>
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      {/* ── Sticky input at bottom ── */}
      <div className="shrink-0 border-t border-gray-100 bg-white/80 backdrop-blur-md px-4 py-3">
        <InputBar />
      </div>
    </div>
  )
}
