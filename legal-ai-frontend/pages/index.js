import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Upload, Send, FileText, AlertTriangle, Scale, PenTool,
  Brain, Shield, Search, ChevronRight, X, CheckCircle,
  Activity, Layers, Terminal, Gavel, Trash2,
  Zap, Eye, BarChart2, RefreshCw, Copy, Download,
  ChevronDown, ChevronUp, Database, Cpu, Wifi, WifiOff, FilePlus
} from 'lucide-react'

const API = 'http://localhost:5000'

const AGENTS = [
  { id: 'Legal Simplifier',   label: 'Simplifier',  icon: Search,        color: '#00D4FF', bg: 'rgba(0,212,255,0.08)',  desc: 'Clause explanation & plain english' },
  { id: 'Risk Detector',      label: 'Risk',         icon: AlertTriangle, color: '#FF8040', bg: 'rgba(255,128,64,0.08)', desc: 'Contract risk identification' },
  { id: 'Summons Handler',    label: 'Summons',      icon: Gavel,         color: '#C9A84C', bg: 'rgba(201,168,76,0.08)', desc: 'Court doc parsing & deadlines' },
  { id: 'Response Generator', label: 'Drafter',      icon: PenTool,       color: '#00CC88', bg: 'rgba(0,204,136,0.08)',  desc: 'Legal response drafting' },
]

const INTENT_COLORS = {
  simplify: { color: '#00D4FF', bg: 'rgba(0,212,255,0.1)', border: 'rgba(0,212,255,0.3)' },
  risk:     { color: '#FF8040', bg: 'rgba(255,128,64,0.1)', border: 'rgba(255,128,64,0.3)' },
  summons:  { color: '#C9A84C', bg: 'rgba(201,168,76,0.1)', border: 'rgba(201,168,76,0.3)' },
  draft:    { color: '#00CC88', bg: 'rgba(0,204,136,0.1)', border: 'rgba(0,204,136,0.3)' },
  general:  { color: '#7A8FA8', bg: 'rgba(122,143,168,0.1)', border: 'rgba(122,143,168,0.3)' },
}

const SUGGESTIONS = [
  { text: 'What are the main risks in this contract?', intent: 'risk' },
  { text: 'Explain the indemnification clause in plain English', intent: 'simplify' },
  { text: 'Who are the parties and what are the court deadlines?', intent: 'summons' },
  { text: 'Draft a formal response to the summons', intent: 'draft' },
  { text: 'Is the IP assignment clause fair?', intent: 'risk' },
  { text: 'What happens if they terminate the contract early?', intent: 'simplify' },
]

// ─── Small helpers ──────────────────────────────────────────────────────────

function RiskPill({ level }) {
  return (
    <span className={`risk-pill risk-${level?.toLowerCase()}`}>
      {level}
    </span>
  )
}

function IntentPill({ intent }) {
  const c = INTENT_COLORS[intent] || INTENT_COLORS.general
  return (
    <span className="intent-pill" style={{ color: c.color, background: c.bg, borderColor: c.border, fontSize: 10.5, padding: '2px 10px', borderRadius: 20, fontFamily: "'JetBrains Mono',monospace" }}>
      {intent}
    </span>
  )
}

function CopyBtn({ text }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1500) }}
      style={{ background: 'var(--surface2)', border: '1px solid var(--border2)', borderRadius: 6, padding: '4px 10px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, color: copied ? 'var(--green)' : 'var(--text3)', fontFamily: "'JetBrains Mono',monospace", fontSize: 11, transition: 'all 0.2s' }}>
      <Copy size={11} /> {copied ? 'copied' : 'copy'}
    </button>
  )
}

// ─── Agent Activity Panel ────────────────────────────────────────────────────

function AgentPanel({ states }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {AGENTS.map(agent => {
        const state = states[agent.id] || 'idle'
        const Icon = agent.icon
        const isActive = state !== 'idle'
        return (
          <div key={agent.id} className={`agent-node state-${state}`} style={{ padding: '12px 14px', display: 'flex', alignItems: 'center', gap: 12 }}>
            {/* Icon */}
            <div style={{
              width: 34, height: 34, borderRadius: 8, flexShrink: 0,
              background: isActive ? agent.bg : 'rgba(26,40,58,0.5)',
              border: `1px solid ${isActive ? agent.color + '40' : 'var(--border)'}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 0.4s'
            }}>
              <Icon size={14} style={{ color: isActive ? agent.color : 'var(--text3)', transition: 'color 0.4s' }} />
            </div>

            {/* Label */}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11.5, color: isActive ? agent.color : 'var(--text3)', letterSpacing: '0.04em', transition: 'color 0.4s' }}>
                {agent.label.toUpperCase()}
              </div>
              <div style={{ fontSize: 10.5, color: 'var(--text3)', marginTop: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {agent.desc}
              </div>
            </div>

            {/* State indicator */}
            <div style={{ flexShrink: 0 }}>
              {state === 'idle' && (
                <div style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--border2)' }} />
              )}
              {state === 'running' && (
                <div style={{ display: 'flex', gap: 3 }}>
                  {[0, 1, 2].map(i => (
                    <div key={i} className="dot" style={{ width: 4, height: 4, animationDelay: `${i * 0.16}s` }} />
                  ))}
                </div>
              )}
              {state === 'done' && (
                <CheckCircle size={14} style={{ color: agent.color }} />
              )}
              {state === 'error' && (
                <X size={14} style={{ color: 'var(--red)' }} />
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ─── Result Cards ────────────────────────────────────────────────────────────

function SynthesisCard({ response }) {
  const [expanded, setExpanded] = useState({})
  const toggle = key => setExpanded(p => ({ ...p, [key]: !p[key] }))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Agent row */}
      {response.agents_used?.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', paddingBottom: 14, borderBottom: '1px solid var(--border)' }}>
          <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10.5, color: 'var(--text3)', letterSpacing: '0.06em' }}>TRIGGERED</span>
          {response.agents_used.map(name => {
            const a = AGENTS.find(x => x.id === name)
            const Icon = a?.icon || Brain
            return (
              <span key={name} style={{
                display: 'inline-flex', alignItems: 'center', gap: 6,
                fontFamily: "'JetBrains Mono',monospace", fontSize: 11,
                color: a?.color || 'var(--text2)',
                background: a?.bg || 'var(--surface2)',
                border: `1px solid ${a?.color ? a.color + '30' : 'var(--border2)'}`,
                borderRadius: 6, padding: '3px 10px'
              }}>
                <Icon size={11} /> {name}
              </span>
            )
          })}
          <div style={{ flex: 1 }} />
          {response.intents?.map(i => <IntentPill key={i} intent={i} />)}
        </div>
      )}

      {/* Synthesis */}
      {response.synthesis && (
        <div style={{ background: 'var(--surface2)', border: '1px solid var(--border2)', borderRadius: 12, padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <Layers size={13} style={{ color: 'var(--gold)' }} />
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: 'var(--gold)', letterSpacing: '0.06em' }}>AI SYNTHESIS</span>
            </div>
            <CopyBtn text={response.synthesis} />
          </div>
          <p style={{ fontSize: 14, lineHeight: 1.8, color: 'var(--text)', whiteSpace: 'pre-wrap' }}>{response.synthesis}</p>
        </div>
      )}

      {/* Per-agent details */}
      {Object.entries(response.results || {}).map(([key, res]) => {
        const agent = AGENTS.find(a => a.id === res.agent) || {}
        const Icon = agent.icon || Brain
        const isOpen = expanded[key] !== false
        return (
          <div key={key} style={{ border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden' }}>
            {/* Header */}
            <button
              onClick={() => toggle(key)}
              style={{ width: '100%', background: 'var(--surface)', border: 'none', padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', borderBottom: isOpen ? '1px solid var(--border)' : 'none' }}>
              <div style={{ width: 28, height: 28, borderRadius: 7, background: agent.bg || 'var(--surface2)', border: `1px solid ${agent.color ? agent.color + '30' : 'var(--border2)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Icon size={13} style={{ color: agent.color || 'var(--text2)' }} />
              </div>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12, color: agent.color || 'var(--text2)', letterSpacing: '0.04em', flex: 1, textAlign: 'left' }}>
                {res.agent?.toUpperCase()}
              </span>
              {res.error && <span style={{ fontSize: 11, color: 'var(--red)' }}>ERROR</span>}
              {isOpen ? <ChevronUp size={13} style={{ color: 'var(--text3)' }} /> : <ChevronDown size={13} style={{ color: 'var(--text3)' }} />}
            </button>

            {/* Body */}
            {isOpen && (
              <div style={{ padding: 16, background: 'var(--surface)' }}>
                {res.error ? (
                  <div style={{ color: 'var(--red)', fontSize: 13 }}>{res.error}</div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {/* Main answer */}
                    {res.answer && (
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
                          <CopyBtn text={res.answer} />
                        </div>
                        <p style={{ fontSize: 13.5, lineHeight: 1.8, color: 'var(--text2)', whiteSpace: 'pre-wrap' }}>{res.answer}</p>
                      </div>
                    )}

                    {/* Summons parsed data */}
                    {res.parsed && (
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                        {Object.entries(res.parsed).filter(([, v]) => v && typeof v !== 'object').map(([k, v]) => (
                          <div key={k} style={{ background: 'var(--surface2)', borderRadius: 8, padding: '8px 12px' }}>
                            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 9.5, color: 'var(--text3)', letterSpacing: '0.07em', marginBottom: 3 }}>{k.toUpperCase().replace(/_/g, ' ')}</div>
                            <div style={{ fontSize: 12.5, color: 'var(--text)', fontWeight: 500 }}>
                              {typeof v === 'boolean' ? (v ? 'Yes' : 'No') : String(v)}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Clauses used */}
                    {res.clauses_used?.length > 0 && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: 'var(--text3)', letterSpacing: '0.07em' }}>SOURCE CLAUSES</div>
                        {res.clauses_used.slice(0, 2).map((c, i) => (
                          <div key={i} style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, padding: 12 }}>
                            {c.legal_types?.length > 0 && (
                              <div style={{ display: 'flex', gap: 4, marginBottom: 6 }}>
                                {c.legal_types.map(t => (
                                  <span key={t} style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 9.5, color: 'var(--gold)', background: 'rgba(201,168,76,0.08)', border: '1px solid rgba(201,168,76,0.2)', borderRadius: 4, padding: '1px 6px' }}>{t}</span>
                                ))}
                              </div>
                            )}
                            <p style={{ fontSize: 12, color: 'var(--text3)', lineHeight: 1.7 }}>{c.text}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ─── Message Bubble ─────────────────────────────────────────────────────────

function MessageBubble({ msg, idx }) {
  const isUser = msg.role === 'user'
  return (
    <div
      className="anim-slide-up"
      style={{ animationDelay: `${idx * 0.03}s`, display: 'flex', flexDirection: 'column', gap: 4, alignItems: isUser ? 'flex-end' : 'flex-start' }}>

      {/* Role label */}
      <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: 'var(--text3)', letterSpacing: '0.06em', paddingInline: 4 }}>
        {isUser ? 'YOU' : 'LEXAI'}
        <span style={{ marginLeft: 8, opacity: 0.5 }}>
          {msg.timestamp.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>

      {/* Bubble */}
      {isUser ? (
        <div style={{ maxWidth: '78%', background: 'linear-gradient(135deg, rgba(201,168,76,0.12), rgba(201,168,76,0.06))', border: '1px solid rgba(201,168,76,0.2)', borderRadius: '14px 14px 4px 14px', padding: '12px 16px' }}>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: 'var(--text)' }}>{msg.content}</p>
        </div>
      ) : (
        <div style={{ width: '100%' }}>
          {msg.loading ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 14, padding: '14px 18px' }}>
              <div style={{ display: 'flex', gap: 5 }}>
                <div className="dot" />
                <div className="dot" />
                <div className="dot" />
              </div>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11.5, color: 'var(--text3)' }}>
                {msg.loadingText || 'Processing...'}
              </span>
            </div>
          ) : (
            <div className="card" style={{ padding: 18 }}>
              {msg.response ? (
                <SynthesisCard response={msg.response} />
              ) : (
                <p style={{ fontSize: 14, lineHeight: 1.7, color: 'var(--text2)' }}>{msg.content}</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Upload Panel ────────────────────────────────────────────────────────────

function UploadPanel({ onUploaded, docStatus }) {
  const [dragging, setDragging] = useState(false)
  const [files, setFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const inputRef = useRef()

  const doUpload = async (fileList) => {
    setUploading(true)
    setResult(null)
    try {
      const fd = new FormData()
      for (const f of fileList) fd.append('files', f)
      const res = await fetch(`${API}/upload`, { method: 'POST', body: fd })
      const data = await res.json()
      setResult(data)
      if (data.success) onUploaded()
    } catch (e) {
      setResult({ success: false, error: e.message })
    }
    setUploading(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const dropped = Array.from(e.dataTransfer.files)
    setFiles(dropped)
    doUpload(dropped)
  }

  const handleSelect = (e) => {
    const selected = Array.from(e.target.files)
    setFiles(selected)
    doUpload(selected)
  }

  const clearDB = async () => {
    try {
      await fetch(`${API}/documents`, { method: 'DELETE' })
      setResult(null)
      setFiles([])
      onUploaded()
    } catch (e) { console.error(e) }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Status line */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Database size={13} style={{ color: docStatus.ready ? 'var(--gold)' : 'var(--text3)' }} />
          <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: 'var(--text3)' }}>
            {docStatus.ready
              ? <span><span style={{ color: 'var(--gold)' }}>{docStatus.clauseCount}</span> clauses indexed</span>
              : 'No documents loaded'}
          </span>
        </div>
        {docStatus.ready && (
          <button className="btn-outline" onClick={clearDB} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '5px 10px' }}>
            <Trash2 size={11} /> Clear DB
          </button>
        )}
      </div>

      {/* Drop zone */}
      <div
        className={`upload-zone${dragging ? ' dragging' : ''}`}
        style={{ padding: 28, textAlign: 'center', cursor: 'pointer' }}
        onDragEnter={e => { e.preventDefault(); setDragging(true) }}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}>
        <input ref={inputRef} type="file" multiple accept=".pdf,.docx,.txt" style={{ display: 'none' }} onChange={handleSelect} />

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 52, height: 52, borderRadius: 14, background: 'var(--surface2)', border: '1px solid var(--border2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {uploading
              ? <RefreshCw size={22} style={{ color: 'var(--gold)', animation: 'spin-slow 1s linear infinite' }} />
              : <Upload size={22} style={{ color: dragging ? 'var(--cyan)' : 'var(--gold)' }} />
            }
          </div>
          <div>
            <div style={{ fontSize: 13.5, color: uploading ? 'var(--gold)' : 'var(--text)', fontWeight: 500, marginBottom: 4 }}>
              {uploading ? 'Processing document...' : dragging ? 'Drop to upload' : 'Drop files here or click to browse'}
            </div>
            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10.5, color: 'var(--text3)' }}>PDF · DOCX · TXT</div>
          </div>
        </div>
      </div>

      {/* Current files */}
      {docStatus.docs?.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          {docStatus.docs.map((doc, i) => {
            const name = doc.split('/').pop().split('\\').pop()
            return (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 9, background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, padding: '7px 12px' }}>
                <FileText size={13} style={{ color: 'var(--gold)', flexShrink: 0 }} />
                <span style={{ fontSize: 12.5, color: 'var(--text2)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name}</span>
                <CheckCircle size={12} style={{ color: 'var(--green)', flexShrink: 0 }} />
              </div>
            )
          })}
        </div>
      )}

      {/* Upload result */}
      {result && (
        <div style={{ borderRadius: 10, padding: '10px 14px', background: result.success ? 'rgba(0,204,136,0.07)' : 'rgba(255,69,69,0.07)', border: `1px solid ${result.success ? 'rgba(0,204,136,0.25)' : 'rgba(255,69,69,0.25)'}` }}>
          <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: result.success ? 'var(--green)' : 'var(--red)' }}>
            {result.success ? `✓ ${result.files_uploaded} file(s) uploaded · ${result.clause_count} clauses indexed` : `✗ ${result.error}`}
          </div>
          {result.details?.map((d, i) => (
            <div key={i} style={{ fontSize: 11, color: d.success ? 'var(--text3)' : 'var(--red)', marginTop: 3 }}>
              {d.file}: {d.success ? `${d.clauses_added} clauses` : d.error}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Main page ───────────────────────────────────────────────────────────────

export default function Home() {
  const [messages, setMessages] = useState([])
  const [query, setQuery] = useState('')
  const [tone, setTone] = useState('formal')
  const [agentStates, setAgentStates] = useState({})
  const [loading, setLoading] = useState(false)
  const [serverStatus, setServerStatus] = useState(null)
  const [docStatus, setDocStatus] = useState({ ready: false, clauseCount: 0, docs: [] })
  const [sidebarTab, setSidebarTab] = useState('upload') // 'upload' | 'agents' | 'info'
  const messagesEndRef = useRef()
  const textareaRef = useRef()

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Fetch server status
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API}/status`)
      const data = await res.json()
      setServerStatus(data)
      setDocStatus({ ready: data.db_ready, clauseCount: data.clause_count, docs: data.documents || [] })
    } catch {
      setServerStatus(null)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const t = setInterval(fetchStatus, 8000)
    return () => clearInterval(t)
  }, [fetchStatus])

  const resetAgents = () => setAgentStates({})

  const send = async () => {
    if (!query.trim() || loading) return
    const q = query.trim()
    setQuery('')
    setLoading(true)
    resetAgents()

    // User message
    const uid = Date.now().toString()
    setMessages(prev => [...prev, { id: uid, role: 'user', content: q, timestamp: new Date() }])

    // Loading placeholder
    const lid = uid + '_loading'
    setMessages(prev => [...prev, { id: lid, role: 'assistant', loading: true, loadingText: 'Classifying intent...', timestamp: new Date() }])

    try {
      // Set all to running state briefly for effect
      setMessages(prev => prev.map(m => m.id === lid ? { ...m, loadingText: 'Dispatching agents...' } : m))

      const res = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, tone })
      })
      const data = await res.json()

      // Animate agents that were used
      const usedAgents = data.agents_used || []
      usedAgents.forEach(name => setAgentStates(prev => ({ ...prev, [name]: 'running' })))
      setMessages(prev => prev.map(m => m.id === lid ? { ...m, loadingText: `Running ${usedAgents.join(', ')}...` } : m))

      await new Promise(r => setTimeout(r, 600))
      usedAgents.forEach(name => setAgentStates(prev => ({ ...prev, [name]: 'done' })))

      // Replace loading with real response
      setMessages(prev => prev.map(m =>
        m.id === lid
          ? { id: lid, role: 'assistant', content: data.synthesis || data.error || 'No response.', response: data.success ? data : null, timestamp: new Date() }
          : m
      ))

      if (data.success) {
        setSidebarTab('agents')
        await fetchStatus()
      }
    } catch (e) {
      setMessages(prev => prev.map(m =>
        m.id === lid
          ? { id: lid, role: 'assistant', content: `Connection error: ${e.message}`, timestamp: new Date() }
          : m
      ))
    }
    setLoading(false)
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  const online = serverStatus?.status === 'online'

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="grid-bg" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* ── Top bar ── */}
      <header style={{ background: 'rgba(7,9,15,0.9)', backdropFilter: 'blur(12px)', borderBottom: '1px solid var(--border)', padding: '0 28px', height: 58, display: 'flex', alignItems: 'center', gap: 20, position: 'sticky', top: 0, zIndex: 100 }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginRight: 8 }}>
          <div style={{ width: 32, height: 32, borderRadius: 9, background: 'linear-gradient(135deg, rgba(201,168,76,0.2), rgba(201,168,76,0.05))', border: '1px solid rgba(201,168,76,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Scale size={16} style={{ color: 'var(--gold)' }} />
          </div>
          <div>
            <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 18, color: 'var(--gold)', lineHeight: 1 }}>LexAI</div>
            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 9, color: 'var(--text3)', letterSpacing: '0.1em' }}>LEGAL INTELLIGENCE</div>
          </div>
        </div>

        {/* Sep */}
        <div style={{ width: 1, height: 28, background: 'var(--border)' }} />

        {/* Nav pills */}
        <div style={{ display: 'flex', gap: 4 }}>
          {[
            { label: 'ANALYSIS', icon: Brain },
            { label: 'RISK', icon: Shield },
            { label: 'DOCUMENTS', icon: FileText },
          ].map(({ label, icon: Icon }) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 12px', borderRadius: 7, fontFamily: "'JetBrains Mono',monospace", fontSize: 10.5, color: 'var(--text3)', cursor: 'default', letterSpacing: '0.05em' }}>
              <Icon size={11} /> {label}
            </div>
          ))}
        </div>

        <div style={{ flex: 1 }} />

        {/* Server status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, fontFamily: "'JetBrains Mono',monospace", fontSize: 11 }}>
          {serverStatus && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text3)' }}>
                <Cpu size={11} style={{ color: 'var(--cyan)' }} />
                <span style={{ color: 'var(--text3)' }}>{serverStatus.model?.split('-').slice(0, 2).join('-')}</span>
              </div>
              <div style={{ width: 1, height: 16, background: 'var(--border)' }} />
              <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                <Database size={11} style={{ color: docStatus.ready ? 'var(--gold)' : 'var(--text3)' }} />
                <span style={{ color: docStatus.ready ? 'var(--gold)' : 'var(--text3)' }}>
                  {docStatus.ready ? `${docStatus.clauseCount} clauses` : 'No DB'}
                </span>
              </div>
              <div style={{ width: 1, height: 16, background: 'var(--border)' }} />
            </>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: online ? 'var(--green)' : 'var(--red)', boxShadow: online ? '0 0 8px var(--green)' : 'none', animation: online ? 'blink 2s infinite' : 'none' }} />
            {online ? <Wifi size={12} style={{ color: 'var(--green)' }} /> : <WifiOff size={12} style={{ color: 'var(--red)' }} />}
            <span style={{ color: online ? 'var(--green)' : 'var(--red)' }}>{online ? 'ONLINE' : 'OFFLINE'}</span>
          </div>
        </div>
      </header>

      {/* ── Main layout ── */}
      <div style={{ flex: 1, display: 'flex', height: 'calc(100vh - 58px)' }}>

        {/* ── Left sidebar — fixed height, only inner content scrolls ── */}
        <aside style={{ width: 300, flexShrink: 0, borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', background: 'var(--bg2)', position: 'sticky', top: 0 }}>

          {/* Sidebar tabs */}
          <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', background: 'var(--bg)' }}>
            {[
              { id: 'upload', label: 'UPLOAD', icon: Upload },
              { id: 'agents', label: 'AGENTS', icon: Activity },
              { id: 'info',   label: 'ABOUT',  icon: Eye },
            ].map(tab => {
              const Icon = tab.icon
              const active = sidebarTab === tab.id
              return (
                <button
                  key={tab.id}
                  onClick={() => setSidebarTab(tab.id)}
                  style={{
                    flex: 1, background: 'none', border: 'none',
                    borderBottom: active ? '2px solid var(--gold)' : '2px solid transparent',
                    padding: '12px 6px',
                    cursor: 'pointer',
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
                    color: active ? 'var(--gold)' : 'var(--text3)',
                    transition: 'all 0.2s'
                  }}>
                  <Icon size={13} />
                  <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 9, letterSpacing: '0.06em' }}>{tab.label}</span>
                </button>
              )
            })}
          </div>

          {/* Sidebar content */}
          <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
            {sidebarTab === 'upload' && (
              <UploadPanel onUploaded={fetchStatus} docStatus={docStatus} />
            )}

            {sidebarTab === 'agents' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {/* Orchestrator box */}
                <div style={{ background: 'var(--surface)', border: '1px solid rgba(201,168,76,0.2)', borderRadius: 12, padding: '14px 14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                    <div style={{ width: 28, height: 28, borderRadius: 8, background: 'rgba(201,168,76,0.1)', border: '1px solid rgba(201,168,76,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Brain size={14} style={{ color: 'var(--gold)' }} />
                    </div>
                    <div>
                      <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: 'var(--gold)', letterSpacing: '0.04em' }}>ORCHESTRATOR</div>
                      <div style={{ fontSize: 10.5, color: 'var(--text3)' }}>Central coordinator</div>
                    </div>
                    <div style={{ marginLeft: 'auto', width: 8, height: 8, borderRadius: '50%', background: loading ? 'var(--cyan)' : 'var(--border2)', boxShadow: loading ? '0 0 10px var(--cyan)' : 'none', transition: 'all 0.3s' }} />
                  </div>
                  <div style={{ height: 1, background: 'var(--border)', marginBottom: 10 }} />
                  {/* Flow arrows */}
                  <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: 'var(--text3)', lineHeight: 1.9 }}>
                    <div style={{ color: loading ? 'var(--cyan)' : 'var(--text3)', transition: 'color 0.3s' }}>① Receive query</div>
                    <div style={{ color: loading ? 'var(--gold)' : 'var(--text3)', transition: 'color 0.3s', transitionDelay: '0.1s' }}>② Classify intent</div>
                    <div style={{ color: loading ? 'var(--gold)' : 'var(--text3)', transition: 'color 0.3s', transitionDelay: '0.2s' }}>③ Dispatch agents</div>
                    <div style={{ color: loading ? 'var(--green)' : 'var(--text3)', transition: 'color 0.3s', transitionDelay: '0.3s' }}>④ Synthesize results</div>
                  </div>
                </div>

                {/* Agent nodes */}
                <div>
                  <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: 'var(--text3)', letterSpacing: '0.07em', marginBottom: 8 }}>AGENT STATUS</div>
                  <AgentPanel states={agentStates} />
                </div>

                {/* Reset */}
                {Object.keys(agentStates).length > 0 && (
                  <button className="btn-outline" onClick={resetAgents} style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
                    <RefreshCw size={11} /> Reset States
                  </button>
                )}
              </div>
            )}

            {sidebarTab === 'info' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 20, color: 'var(--gold)', lineHeight: 1.3 }}>Legal AI System</div>
                <p style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.8 }}>
                  An intelligent multi-agent system that analyzes legal documents using specialized AI agents coordinated by a central orchestrator.
                </p>
                <div style={{ height: 1, background: 'var(--border)' }} />
                {[
                  { label: 'Backend', value: 'Flask + Python' },
                  { label: 'LLM', value: 'Groq / Llama-3.3-70b' },
                  { label: 'Embeddings', value: 'LegalBERT' },
                  { label: 'Vector DB', value: 'FAISS' },
                  { label: 'Frontend', value: 'Next.js + JS' },
                ].map(({ label, value }) => (
                  <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
                    <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: 'var(--text3)' }}>{label}</span>
                    <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: 'var(--gold)' }}>{value}</span>
                  </div>
                ))}
                <div style={{ height: 1, background: 'var(--border)' }} />
                <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: 'var(--text3)' }}>API ENDPOINTS</div>
                {['POST /upload', 'POST /ask', 'GET /status', 'DELETE /documents'].map(ep => (
                  <div key={ep} style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: 'var(--cyan)', background: 'rgba(0,212,255,0.05)', border: '1px solid rgba(0,212,255,0.15)', borderRadius: 6, padding: '5px 10px' }}>{ep}</div>
                ))}
              </div>
            )}
          </div>
        </aside>

        {/* ── Chat area — messages scroll, input stays fixed ── */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', minWidth: 0 }}>

          {/* Messages — only this scrolls */}
          <div className="scan-wrap" style={{ flex: 1, overflowY: 'auto', padding: '28px 32px', display: 'flex', flexDirection: 'column', gap: 20 }}>

            {/* Empty state */}
            {messages.length === 0 && (
              <div className="anim-fade-in" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 32, padding: '40px 20px' }}>
                {/* Hero icon */}
                <div style={{ position: 'relative' }}>
                  <div style={{ width: 90, height: 90, borderRadius: '50%', background: 'radial-gradient(circle, rgba(201,168,76,0.15) 0%, transparent 70%)', border: '1px solid rgba(201,168,76,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', animation: 'float 3s ease-in-out infinite' }}>
                    <Scale size={36} style={{ color: 'var(--gold)' }} />
                  </div>
                  {/* Orbiting dots */}
                  {AGENTS.map((a, i) => {
                    const angle = (i / AGENTS.length) * 360
                    const rad = (angle * Math.PI) / 180
                    const r = 60
                    return (
                      <div key={a.id} style={{
                        position: 'absolute',
                        top: '50%', left: '50%',
                        transform: `translate(-50%,-50%) rotate(${angle}deg) translateX(${r}px)`,
                        width: 24, height: 24, borderRadius: '50%',
                        background: a.bg, border: `1px solid ${a.color}40`,
                        display: 'flex', alignItems: 'center', justifyContent: 'center'
                      }}>
                        <a.icon size={11} style={{ color: a.color }} />
                      </div>
                    )
                  })}
                </div>

                <div style={{ textAlign: 'center' }}>
                  <h1 className="shimmer-gold" style={{ fontFamily: "'DM Serif Display',serif", fontSize: 34, marginBottom: 10 }}>
                    Legal Intelligence Platform
                  </h1>
                  <p style={{ fontSize: 15, color: 'var(--text2)', maxWidth: 480, lineHeight: 1.7 }}>
                    Upload your legal documents and ask anything. The AI orchestrator will automatically dispatch the right agents to analyze, explain, detect risks, and draft responses.
                  </p>
                </div>

                {/* Suggestion chips */}
                {docStatus.ready && (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, maxWidth: 680, width: '100%' }}>
                    {SUGGESTIONS.map((s, i) => {
                      const c = INTENT_COLORS[s.intent]
                      return (
                        <button
                          key={i}
                          onClick={() => setQuery(s.text)}
                          style={{
                            background: 'var(--surface)', border: '1px solid var(--border)',
                            borderRadius: 10, padding: '11px 14px',
                            cursor: 'pointer', textAlign: 'left', transition: 'all 0.2s',
                            display: 'flex', alignItems: 'flex-start', gap: 10
                          }}
                          onMouseEnter={e => { e.currentTarget.style.borderColor = c.border; e.currentTarget.style.background = c.bg }}
                          onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.background = 'var(--surface)' }}>
                          <ChevronRight size={13} style={{ color: c.color, marginTop: 1, flexShrink: 0 }} />
                          <span style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.5 }}>{s.text}</span>
                        </button>
                      )
                    })}
                  </div>
                )}

                {!docStatus.ready && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'rgba(201,168,76,0.06)', border: '1px solid rgba(201,168,76,0.2)', borderRadius: 10, padding: '12px 18px' }}>
                    <Upload size={16} style={{ color: 'var(--gold)', flexShrink: 0 }} />
                    <span style={{ fontSize: 13.5, color: 'var(--text2)' }}>Upload a document from the left panel to begin analysis.</span>
                  </div>
                )}
              </div>
            )}

            {/* Messages */}
            {messages.map((msg, i) => (
              <MessageBubble key={msg.id} msg={msg} idx={i} />
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* ── Input bar — sticky at bottom, never scrolls ── */}
          <div style={{ borderTop: '1px solid var(--border)', padding: '16px 28px 20px', background: 'rgba(7,9,15,0.95)', backdropFilter: 'blur(16px)', flexShrink: 0, zIndex: 10 }}>

            {/* Tone selector + suggestion row */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10.5, color: 'var(--text3)' }}>TONE:</span>
              {['formal', 'assertive', 'conciliatory'].map(t => (
                <button
                  key={t}
                  onClick={() => setTone(t)}
                  style={{
                    fontFamily: "'JetBrains Mono',monospace", fontSize: 10.5, letterSpacing: '0.04em',
                    padding: '3px 10px', borderRadius: 6, border: '1px solid', cursor: 'pointer',
                    borderColor: tone === t ? 'var(--gold)' : 'var(--border2)',
                    color: tone === t ? 'var(--gold)' : 'var(--text3)',
                    background: tone === t ? 'rgba(201,168,76,0.08)' : 'transparent',
                    transition: 'all 0.2s'
                  }}>
                  {t}
                </button>
              ))}

              <div style={{ flex: 1 }} />

              {/* Quick prompts */}
              <div style={{ display: 'flex', gap: 6 }}>
                {[
                  { label: '⚡ Risks', text: 'Identify all risks in this document' },
                  { label: '📋 Summary', text: 'Summarize this legal document' },
                  { label: '⚖️ Draft', text: 'Draft a formal reply to this summons' },
                ].map(({ label, text }) => (
                  <button key={label} className="btn-outline" onClick={() => setQuery(text)} style={{ fontSize: 11, padding: '4px 10px' }}>
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Textarea + send */}
            <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
              <div style={{ flex: 1, position: 'relative' }}>
                <textarea
                  ref={textareaRef}
                  className="chat-input"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyDown={handleKey}
                  placeholder="Ask about your document — explain a clause, find risks, analyze summons, or draft a response..."
                  rows={2}
                  style={{ maxHeight: 160 }}
                />
              </div>

              <button
                className="btn-primary"
                onClick={send}
                disabled={!query.trim() || loading || !docStatus.ready}
                style={{ height: 52, width: 52, padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                {loading
                  ? <RefreshCw size={18} style={{ animation: 'spin-slow 1s linear infinite' }} />
                  : <Send size={18} />
                }
              </button>
            </div>

            {/* Bottom status line */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 8, fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: 'var(--text3)' }}>
              <span>Enter to send · Shift+Enter for newline</span>
              <div style={{ flex: 1 }} />
              {loading && (
                <span style={{ color: 'var(--cyan)', display: 'flex', alignItems: 'center', gap: 5 }}>
                  <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--cyan)', animation: 'blink 0.8s infinite' }} />
                  PROCESSING
                </span>
              )}
              {!docStatus.ready && <span style={{ color: 'var(--red)' }}>Upload a document first</span>}
              {docStatus.ready && !loading && <span style={{ color: 'var(--green)' }}>READY</span>}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}


