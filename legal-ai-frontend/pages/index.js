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

function DownloadPDFBtn({ text, filename = 'response.pdf' }) {
  const handleDownload = () => {
    // Create a printable window with the content formatted as PDF
    const win = window.open('', '_blank')
    const htmlContent = text
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\n(Note|NOTE):.*(\n|$)/g, '')   // strip Note: lines
      .replace(/\n/g, '<br/>')
    win.document.write(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>Legal Response</title>
        <style>
          body { font-family: Georgia, serif; font-size: 13pt; line-height: 1.8; color: #1a1a1a; max-width: 800px; margin: 40px auto; padding: 0 40px; }
          strong { font-weight: bold; }
          @page { margin: 20mm; size: A4; }
          @media print {
            body { margin: 0; }
            html { -webkit-print-color-adjust: exact; }
          }
        </style>
      </head>
      <body>
        <div>${htmlContent}</div>
        <script>
          window.onload = () => {
            // Remove browser default header/footer via title trick
            document.title = '';
            window.print();
          };
        <\/script>
      </body>
      </html>
    `)
    win.document.close()
  }
  return (
    <button
      onClick={handleDownload}
      style={{ background: 'rgba(0,204,136,0.1)', border: '1px solid rgba(0,204,136,0.3)', borderRadius: 6, padding: '4px 10px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, color: 'var(--green)', fontFamily: "'JetBrains Mono',monospace", fontSize: 11, transition: 'all 0.2s' }}>
      <Download size={11} /> download pdf
    </button>
  )
}



function RichText({ text, style = {} }) {
  if (!text) return null
  // Split on **...** patterns and render bold segments
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return (
    <span style={style}>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={i}>{part.slice(2, -2)}</strong>
        }
        return <span key={i}>{part}</span>
      })}
    </span>
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
          <p style={{ fontSize: 14, lineHeight: 1.8, color: 'var(--text)', whiteSpace: 'pre-wrap' }}><RichText text={response.synthesis} /></p>
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
                        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8, gap: 8 }}>
                          {res.agent === 'Response Generator' && (
                            <DownloadPDFBtn text={res.answer} filename="legal-response.pdf" />
                          )}
                          <CopyBtn text={res.answer} />
                        </div>
                        <p style={{ fontSize: 13.5, lineHeight: 1.8, color: 'var(--text2)', whiteSpace: 'pre-wrap' }}><RichText text={res.answer} /></p>
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

function ScanProgressBar({ scan }) {
  if (!scan.running && !scan.done) return null
  const pct = scan.percent || 0
  const isRunning = scan.running

  return (
    <div style={{ borderRadius: 10, overflow: 'hidden', border: `1px solid ${isRunning ? 'rgba(255,128,64,0.3)' : 'rgba(0,204,136,0.3)'}`, background: isRunning ? 'rgba(255,128,64,0.05)' : 'rgba(0,204,136,0.05)' }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '9px 12px', borderBottom: `1px solid ${isRunning ? 'rgba(255,128,64,0.2)' : 'rgba(0,204,136,0.2)'}` }}>
        <AlertTriangle size={12} style={{ color: isRunning ? 'var(--orange)' : 'var(--green)', flexShrink: 0 }} />
        <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10.5, color: isRunning ? 'var(--orange)' : 'var(--green)', letterSpacing: '0.05em', flex: 1 }}>
          {isRunning ? 'RISK SCAN IN PROGRESS' : 'RISK SCAN COMPLETE'}
        </span>
        {isRunning && (
          <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: 'var(--text3)' }}>
            {scan.current}/{scan.total}
          </span>
        )}
        {!isRunning && scan.done && (
          <CheckCircle size={12} style={{ color: 'var(--green)' }} />
        )}
      </div>

      {/* Progress bar */}
      <div style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ height: 5, borderRadius: 3, background: 'var(--border2)', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${pct}%`,
            borderRadius: 3,
            background: isRunning
              ? 'linear-gradient(90deg, var(--orange), var(--gold))'
              : 'linear-gradient(90deg, var(--green), #00FF99)',
            transition: 'width 0.6s ease',
            boxShadow: isRunning ? '0 0 8px rgba(255,128,64,0.5)' : '0 0 8px rgba(0,204,136,0.5)',
          }} />
        </div>

        {/* Stats row */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: 'var(--text3)' }}>
            {isRunning
              ? `Analysing clause ${scan.current} of ${scan.total}...`
              : `${scan.total} clauses analysed${scan.failed > 0 ? ` · ${scan.failed} failed` : ''}`
            }
          </span>
          <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: isRunning ? 'var(--orange)' : 'var(--green)', fontWeight: 600 }}>
            {pct}%
          </span>
        </div>

        {/* Executive summary once done */}
        {!isRunning && scan.summary && (
          <div style={{ marginTop: 2, padding: '8px 10px', background: 'rgba(0,204,136,0.06)', borderRadius: 7, border: '1px solid rgba(0,204,136,0.15)' }}>
            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 9.5, color: 'var(--green)', letterSpacing: '0.06em', marginBottom: 4 }}>EXECUTIVE SUMMARY</div>
            <p style={{ fontSize: 11.5, color: 'var(--text2)', lineHeight: 1.7 }}>{scan.summary}</p>
          </div>
        )}
        {scan.error && (
          <div style={{ fontSize: 11, color: 'var(--red)', fontFamily: "'JetBrains Mono',monospace" }}>
            ✗ {scan.error}
          </div>
        )}
      </div>
    </div>
  )
}

function UploadPanel({ onUploaded, docStatus }) {
  const [dragging, setDragging]   = useState(false)
  const [uploading, setUploading] = useState(false)
  const [result, setResult]       = useState(null)
  const [scan, setScan]           = useState({ running: false, done: false, percent: 0, current: 0, total: 0, summary: '', error: '' })
  const inputRef  = useRef()
  const pollRef   = useRef(null)

  // Poll /scan-status while scan is running
  const startPolling = () => {
    if (pollRef.current) return
    pollRef.current = setInterval(async () => {
      try {
        const res  = await fetch(`${API}/scan-status`)
        const data = await res.json()
        setScan(data)
        if (!data.running) {
          clearInterval(pollRef.current)
          pollRef.current = null
          onUploaded()  // refresh status once scan completes
        }
      } catch { /* ignore poll errors */ }
    }, 1500)
  }

  // On mount: check if scan already in progress (e.g. page refresh)
  useEffect(() => {
    const check = async () => {
      try {
        const res  = await fetch(`${API}/scan-status`)
        const data = await res.json()
        setScan(data)
        if (data.running) startPolling()
      } catch {}
    }
    check()
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const doUpload = async (fileList) => {
    setUploading(true)
    setResult(null)
    setScan({ running: false, done: false, percent: 0, current: 0, total: 0, summary: '', error: '' })
    try {
      const fd = new FormData()
      for (const f of fileList) fd.append('files', f)
      const res  = await fetch(`${API}/upload`, { method: 'POST', body: fd })
      const data = await res.json()
      setResult(data)
      if (data.success) {
        onUploaded()
        if (data.scan_started) startPolling()
      }
    } catch (e) {
      setResult({ success: false, error: e.message })
    }
    setUploading(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    doUpload(Array.from(e.dataTransfer.files))
  }

  const handleSelect = (e) => doUpload(Array.from(e.target.files))

  const clearDB = async () => {
    try {
      await fetch(`${API}/documents`, { method: 'DELETE' })
      setResult(null)
      setScan({ running: false, done: false, percent: 0, current: 0, total: 0, summary: '', error: '' })
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
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
        style={{ padding: 24, textAlign: 'center', cursor: uploading ? 'default' : 'pointer' }}
        onDragEnter={e => { e.preventDefault(); setDragging(true) }}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && inputRef.current?.click()}>
        <input ref={inputRef} type="file" multiple accept=".pdf,.docx,.txt" style={{ display: 'none' }} onChange={handleSelect} />
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 46, height: 46, borderRadius: 12, background: 'var(--surface2)', border: '1px solid var(--border2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {uploading
              ? <RefreshCw size={20} style={{ color: 'var(--gold)', animation: 'spin-slow 1s linear infinite' }} />
              : <Upload size={20} style={{ color: dragging ? 'var(--cyan)' : 'var(--gold)' }} />
            }
          </div>
          <div>
            <div style={{ fontSize: 13, color: uploading ? 'var(--gold)' : 'var(--text)', fontWeight: 500, marginBottom: 3 }}>
              {uploading ? 'Ingesting document...' : dragging ? 'Drop to upload' : 'Drop files here or click to browse'}
            </div>
            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: 'var(--text3)' }}>PDF · DOCX · TXT</div>
          </div>
        </div>
      </div>

      {/* Indexed docs */}
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
        <div style={{ borderRadius: 10, padding: '9px 13px', background: result.success ? 'rgba(0,204,136,0.07)' : 'rgba(255,69,69,0.07)', border: `1px solid ${result.success ? 'rgba(0,204,136,0.25)' : 'rgba(255,69,69,0.25)'}` }}>
          <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: result.success ? 'var(--green)' : 'var(--red)' }}>
            {result.success
              ? `✓ ${result.files_uploaded} file(s) uploaded · ${result.clause_count} clauses indexed`
              : `✗ ${result.error}`
            }
          </div>
          {result.details?.map((d, i) => (
            <div key={i} style={{ fontSize: 11, color: d.success ? 'var(--text3)' : 'var(--red)', marginTop: 3 }}>
              {d.file}: {d.success ? `${d.clauses_added} clauses` : d.error}
            </div>
          ))}
          {result.scan_started && (
            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10.5, color: 'var(--orange)', marginTop: 5 }}>
              ⚡ Risk scan started in background...
            </div>
          )}
        </div>
      )}

      {/* Live scan progress bar */}
      <ScanProgressBar scan={scan} />

    </div>
  )
}

// ─── Risk Level Config ────────────────────────────────────────────────────────

const RISK_LEVELS = {
  critical: { color: '#FF3B3B', bg: 'rgba(255,59,59,0.10)', border: 'rgba(255,59,59,0.35)', badge: 'rgba(255,59,59,0.18)', dot: '#FF3B3B', label: 'CRITICAL', emoji: '🚨' },
  high:     { color: '#FF8040', bg: 'rgba(255,128,64,0.10)', border: 'rgba(255,128,64,0.35)', badge: 'rgba(255,128,64,0.18)', dot: '#FF8040', label: 'HIGH',     emoji: '⚠️' },
  medium:   { color: '#F5C518', bg: 'rgba(245,197,24,0.08)', border: 'rgba(245,197,24,0.30)', badge: 'rgba(245,197,24,0.15)', dot: '#F5C518', label: 'MEDIUM',   emoji: '⚡' },
  low:      { color: '#44CC88', bg: 'rgba(68,204,136,0.08)', border: 'rgba(68,204,136,0.25)', badge: 'rgba(68,204,136,0.12)', dot: '#44CC88', label: 'LOW',      emoji: '✓'  },
}

// ─── Risk Panel ───────────────────────────────────────────────────────────────

function RiskPanel({ onClose, onExplain }) {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter]   = useState('all')   // 'all' | 'critical' | 'high' | 'medium' | 'low'
  const [expanded, setExpanded] = useState({})

  useEffect(() => {
    setLoading(true)
    fetch(`${API}/risks`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(() => { setData({ success: false, error: 'Could not connect to server.' }); setLoading(false) })
  }, [])

  const toggle = (i) => setExpanded(p => ({ ...p, [i]: !p[i] }))

  const risks = data?.risks || []
  const visible = filter === 'all' ? risks : risks.filter(r => r.risk_level === filter)
  const counts  = data?.counts || { critical: 0, high: 0, medium: 0, low: 0 }
  const MONO = { fontFamily: "'JetBrains Mono',monospace" }

  return (
    <div style={{
      width: 480, flexShrink: 0,
      borderLeft: '1px solid var(--border)',
      background: 'var(--bg2)',
      display: 'flex', flexDirection: 'column',
      overflow: 'hidden',
      animation: 'slideInRight 0.22s ease',
    }}>
      {/* Header */}
      <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', flexShrink: 0, background: 'rgba(255,128,64,0.04)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <div style={{ width: 30, height: 30, borderRadius: 8, background: 'rgba(255,128,64,0.12)', border: '1px solid rgba(255,128,64,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Shield size={14} style={{ color: 'var(--orange)' }} />
          </div>
          <div>
            <div style={{ ...MONO, fontSize: 12, color: 'var(--orange)', letterSpacing: '0.06em' }}>RISK ANALYSIS</div>
            <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 1 }}>
              {loading ? 'Loading...' : `${data?.total || 0} clauses scanned`}
            </div>
          </div>
          <button onClick={onClose} style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text3)', display: 'flex', padding: 4 }}>
            <X size={16} />
          </button>
        </div>

        {/* Count pills */}
        {!loading && data?.success && (
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {['all', 'critical', 'high', 'medium', 'low'].map(lvl => {
              const c = lvl === 'all' ? { color: 'var(--text2)', border: 'var(--border2)', bg: 'var(--surface2)' } : RISK_LEVELS[lvl]
              const count = lvl === 'all' ? risks.length : counts[lvl] || 0
              const isActive = filter === lvl
              return (
                <button key={lvl} onClick={() => setFilter(lvl)} style={{
                  ...MONO, fontSize: 10, padding: '4px 10px', borderRadius: 20,
                  border: `1px solid ${isActive ? c.color || c.border : 'var(--border)'}`,
                  background: isActive ? (c.bg || 'var(--surface2)') : 'transparent',
                  color: isActive ? (c.color || 'var(--text2)') : 'var(--text3)',
                  cursor: 'pointer', transition: 'all 0.15s', fontWeight: isActive ? 700 : 400,
                }}>
                  {lvl === 'all' ? `ALL (${count})` : `${lvl.toUpperCase()} (${count})`}
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* Risk summary bar */}
      {!loading && data?.success && risks.length > 0 && (
        <div style={{ padding: '10px 20px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
          <div style={{ display: 'flex', height: 6, borderRadius: 4, overflow: 'hidden', gap: 1 }}>
            {['critical', 'high', 'medium', 'low'].map(lvl => {
              const pct = risks.length ? ((counts[lvl] || 0) / risks.length) * 100 : 0
              return pct > 0 ? (
                <div key={lvl} style={{ width: `${pct}%`, background: RISK_LEVELS[lvl].color, transition: 'width 0.4s' }} title={`${lvl}: ${counts[lvl]}`} />
              ) : null
            })}
          </div>
          <div style={{ display: 'flex', gap: 14, marginTop: 6 }}>
            {['critical', 'high', 'medium', 'low'].map(lvl => counts[lvl] > 0 && (
              <div key={lvl} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <div style={{ width: 6, height: 6, borderRadius: '50%', background: RISK_LEVELS[lvl].color }} />
                <span style={{ ...MONO, fontSize: 9.5, color: RISK_LEVELS[lvl].color }}>{counts[lvl]} {lvl}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Executive summary */}
      {!loading && data?.summary && (
        <div style={{ padding: '12px 20px', borderBottom: '1px solid var(--border)', background: 'rgba(255,128,64,0.04)', flexShrink: 0 }}>
          <div style={{ ...MONO, fontSize: 9.5, color: 'var(--orange)', letterSpacing: '0.07em', marginBottom: 5 }}>EXECUTIVE SUMMARY</div>
          <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.7 }}>{data.summary}</p>
        </div>
      )}

      {/* Risk list — BLOCK layout (not flex column) so cards never get height-squished */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px' }}>
        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[1,2,3,4].map(i => (
              <div key={i} style={{ height: 72, borderRadius: 10, background: 'var(--surface)', border: '1px solid var(--border)', animation: 'blink 1.2s infinite', animationDelay: `${i*0.1}s` }} />
            ))}
          </div>
        )}

        {!loading && !data?.success && (
          <div style={{ textAlign: 'center', padding: 40, color: 'var(--red)', fontSize: 13 }}>
            {data?.error || 'Failed to load risks.'}
          </div>
        )}

        {!loading && data?.success && visible.length === 0 && (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <div style={{ fontSize: 28, marginBottom: 10 }}>✓</div>
            <div style={{ color: 'var(--green)', ...MONO, fontSize: 12 }}>
              {risks.length === 0 ? 'No risks scanned yet. Upload a document first.' : `No ${filter} risks found.`}
            </div>
          </div>
        )}

        {!loading && data?.success && visible.map((risk, i) => {
          const lvl   = risk.risk_level || 'low'
          const cfg   = RISK_LEVELS[lvl] || RISK_LEVELS.low
          const open  = expanded[i] !== false  // default open for critical/high
          const types = (risk.risk_types || []).join(', ') || 'general'

          return (
            <div key={`${filter}-${i}`} style={{
              marginBottom: 8,
              border: `1px solid ${cfg.border}`,
              borderRadius: 10,
              background: cfg.bg,
              transition: 'all 0.2s',
            }}>
              {/* Card header */}
              <div
                onClick={() => toggle(i)}
                style={{ padding: '12px 14px', cursor: 'pointer', display: 'flex', alignItems: 'flex-start', gap: 10, minHeight: 48 }}>
                {/* Level badge */}
                <div style={{
                  flexShrink: 0, marginTop: 1,
                  background: cfg.badge,
                  border: `1px solid ${cfg.border}`,
                  borderRadius: 6,
                  padding: '2px 7px',
                  display: 'flex', alignItems: 'center', gap: 4,
                }}>
                  <div style={{ width: 5, height: 5, borderRadius: '50%', background: cfg.color }} />
                  <span style={{ ...MONO, fontSize: 9, color: cfg.color, fontWeight: 700, letterSpacing: '0.06em' }}>{cfg.label}</span>
                </div>

                {/* Summary */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.55, fontWeight: 500, marginBottom: 3 }}>
                    {risk.summary || 'Risk identified'}
                  </p>
                  <div style={{ ...MONO, fontSize: 10, color: 'var(--text3)' }}>{types}</div>
                </div>

                {/* Chevron */}
                {open
                  ? <ChevronUp size={13} style={{ color: 'var(--text3)', flexShrink: 0, marginTop: 2 }} />
                  : <ChevronDown size={13} style={{ color: 'var(--text3)', flexShrink: 0, marginTop: 2 }} />
                }
              </div>

              {/* Expanded body */}
              {open && (
                <div style={{ padding: '12px 14px 14px', borderTop: `1px solid ${cfg.border}`, display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {/* Clause excerpt */}
                  {risk.clause_text && (
                    <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: 7, padding: '8px 11px', border: '1px solid var(--border)' }}>
                      <div style={{ ...MONO, fontSize: 9.5, color: 'var(--text3)', letterSpacing: '0.06em', marginBottom: 5 }}>CLAUSE EXCERPT</div>
                      <p style={{ fontSize: 12, color: 'var(--text3)', lineHeight: 1.65, fontStyle: 'italic' }}>
                        "{(risk.clause_text || '').slice(0, 220)}{risk.clause_text?.length > 220 ? '...' : ''}"
                      </p>
                    </div>
                  )}

                  {/* Recommendation */}
                  {risk.recommendation && (
                    <div style={{ background: 'rgba(0,0,0,0.15)', borderRadius: 7, padding: '8px 11px', border: `1px solid ${cfg.border}` }}>
                      <div style={{ ...MONO, fontSize: 9.5, color: cfg.color, letterSpacing: '0.06em', marginBottom: 5 }}>RECOMMENDATION</div>
                      <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.65 }}>{risk.recommendation}</p>
                    </div>
                  )}

                  {/* Explain button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      // Build a rich grounded query that includes all risk details so the
                      // vector DB has real clause language to search against — not just a label
                      const parts = [
                        `Explain this ${(risk.risk_level || 'legal').toUpperCase()} risk to me in plain English and what it means for me.`,
                        `Risk identified: "${risk.summary}"`,
                      ]
                      if (risk.risk_types?.length > 0) {
                        parts.push(`Risk category: ${risk.risk_types.join(', ')}.`)
                      }
                      if (risk.clause_text) {
                        parts.push(`This risk comes from the following clause in the document: "${risk.clause_text}"`)
                      }
                      if (risk.recommendation) {
                        parts.push(`Suggested action: ${risk.recommendation}`)
                      }
                      parts.push('Please explain: (1) what this clause means in simple language, (2) what real-world risk it creates for me, (3) what could go wrong if I ignore it, and (4) what I should do about it.')
                      onExplain(parts.join(' '))
                    }}
                    style={{
                      alignSelf: 'flex-start',
                      background: cfg.badge, border: `1px solid ${cfg.border}`,
                      borderRadius: 7, padding: '6px 14px', cursor: 'pointer',
                      display: 'flex', alignItems: 'center', gap: 6,
                      color: cfg.color, ...MONO, fontSize: 11, fontWeight: 600,
                      transition: 'all 0.15s',
                    }}
                    onMouseEnter={e => e.currentTarget.style.opacity = '0.75'}
                    onMouseLeave={e => e.currentTarget.style.opacity = '1'}
                  >
                    <Brain size={11} /> Explain this risk
                  </button>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── Analysis Panel ───────────────────────────────────────────────────────────

function AnalysisPanel({ onClose, onAsk }) {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const MONO = { fontFamily: "'JetBrains Mono',monospace" }

  useEffect(() => {
    setLoading(true)
    fetch(`${API}/analysis`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(() => { setData({ success: false, error: 'Could not connect to server.' }); setLoading(false) })
  }, [])

  const totalRisks = data
    ? (data.risk_distribution?.critical || 0) + (data.risk_distribution?.high || 0) +
      (data.risk_distribution?.medium || 0)  + (data.risk_distribution?.low || 0)
    : 0

  const overallRisk = () => {
    if (!data?.risk_distribution) return null
    const { critical, high, medium } = data.risk_distribution
    if (critical > 0) return { label: 'HIGH RISK', color: '#FF3B3B', bg: 'rgba(255,59,59,0.1)' }
    if (high > 2)     return { label: 'ELEVATED RISK', color: '#FF8040', bg: 'rgba(255,128,64,0.1)' }
    if (medium > 3)   return { label: 'MODERATE RISK', color: '#F5C518', bg: 'rgba(245,197,24,0.08)' }
    return              { label: 'LOW RISK', color: '#44CC88', bg: 'rgba(68,204,136,0.08)' }
  }

  const risk = overallRisk()

  const QUICK_QUESTIONS = [
    { text: 'What are the biggest risks I should know about?', intent: 'risk' },
    { text: 'Summarize this document in simple terms', intent: 'simplify' },
    { text: 'What clauses should I negotiate or remove?', intent: 'risk' },
    { text: 'Explain the indemnification and liability clauses', intent: 'simplify' },
  ]

  return (
    <div style={{
      width: 480, flexShrink: 0,
      borderLeft: '1px solid var(--border)',
      background: 'var(--bg2)',
      display: 'flex', flexDirection: 'column',
      overflow: 'hidden',
      animation: 'slideInRight 0.22s ease',
    }}>
      {/* Header */}
      <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', flexShrink: 0, background: 'rgba(0,212,255,0.03)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 30, height: 30, borderRadius: 8, background: 'rgba(0,212,255,0.10)', border: '1px solid rgba(0,212,255,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <BarChart2 size={14} style={{ color: 'var(--cyan)' }} />
          </div>
          <div>
            <div style={{ ...MONO, fontSize: 12, color: 'var(--cyan)', letterSpacing: '0.06em' }}>DOCUMENT ANALYSIS</div>
            <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 1 }}>
              {loading ? 'Loading...' : data?.success ? 'Overview of your document' : 'No document loaded'}
            </div>
          </div>
          <button onClick={onClose} style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text3)', display: 'flex', padding: 4 }}>
            <X size={16} />
          </button>
        </div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 16 }}>
        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[1,2,3].map(i => <div key={i} style={{ height: 90, borderRadius: 10, background: 'var(--surface)', border: '1px solid var(--border)', animation: 'blink 1.2s infinite', animationDelay: `${i*0.15}s` }} />)}
          </div>
        )}

        {!loading && !data?.success && (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Upload size={32} style={{ color: 'var(--text3)', marginBottom: 12 }} />
            <div style={{ color: 'var(--text3)', fontSize: 13 }}>Upload a document to see analysis</div>
          </div>
        )}

        {!loading && data?.success && (
          <>
            {/* Overall verdict */}
            {risk && (
              <div style={{ background: risk.bg, border: `1px solid ${risk.color}44`, borderRadius: 12, padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 14 }}>
                <div style={{ width: 44, height: 44, borderRadius: 10, background: `${risk.color}18`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <AlertTriangle size={20} style={{ color: risk.color }} />
                </div>
                <div>
                  <div style={{ ...MONO, fontSize: 9.5, color: risk.color, letterSpacing: '0.07em', marginBottom: 3 }}>OVERALL ASSESSMENT</div>
                  <div style={{ fontSize: 17, fontWeight: 700, color: risk.color }}>{risk.label}</div>
                  {data.risk_summary && (
                    <p style={{ fontSize: 12, color: 'var(--text2)', marginTop: 5, lineHeight: 1.6 }}>{data.risk_summary}</p>
                  )}
                </div>
              </div>
            )}

            {/* Stats row */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
              {[
                { label: 'CLAUSES', value: data.clause_count, color: 'var(--cyan)', icon: Layers },
                { label: 'RISKS FOUND', value: totalRisks, color: 'var(--orange)', icon: AlertTriangle },
                { label: 'DOCUMENTS', value: data.documents?.length || 0, color: 'var(--gold)', icon: FileText },
              ].map(({ label, value, color, icon: Icon }) => (
                <div key={label} style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: '12px 14px', textAlign: 'center' }}>
                  <Icon size={16} style={{ color, marginBottom: 6 }} />
                  <div style={{ fontSize: 22, fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
                  <div style={{ ...MONO, fontSize: 9, color: 'var(--text3)', marginTop: 4, letterSpacing: '0.05em' }}>{label}</div>
                </div>
              ))}
            </div>

            {/* Docs indexed */}
            {data.documents?.length > 0 && (
              <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: '12px 14px' }}>
                <div style={{ ...MONO, fontSize: 10, color: 'var(--text3)', letterSpacing: '0.07em', marginBottom: 8 }}>INDEXED DOCUMENTS</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                  {data.documents.map((doc, i) => {
                    const name = doc.split('/').pop().split('\\').pop()
                    return (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <FileText size={11} style={{ color: 'var(--gold)', flexShrink: 0 }} />
                        <span style={{ fontSize: 12.5, color: 'var(--text2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name}</span>
                        <CheckCircle size={11} style={{ color: 'var(--green)', flexShrink: 0, marginLeft: 'auto' }} />
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Risk distribution */}
            {totalRisks > 0 && (
              <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: '12px 14px' }}>
                <div style={{ ...MONO, fontSize: 10, color: 'var(--text3)', letterSpacing: '0.07em', marginBottom: 10 }}>RISK DISTRIBUTION</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                  {['critical', 'high', 'medium', 'low'].map(lvl => {
                    const count = data.risk_distribution[lvl] || 0
                    const pct = totalRisks ? (count / totalRisks) * 100 : 0
                    const cfg = RISK_LEVELS[lvl]
                    return (
                      <div key={lvl}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                          <span style={{ ...MONO, fontSize: 10, color: cfg.color }}>{lvl.toUpperCase()}</span>
                          <span style={{ ...MONO, fontSize: 10, color: 'var(--text3)' }}>{count}</span>
                        </div>
                        <div style={{ height: 5, borderRadius: 3, background: 'var(--border2)' }}>
                          <div style={{ height: '100%', width: `${pct}%`, borderRadius: 3, background: cfg.color, transition: 'width 0.5s ease' }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Clause type breakdown */}
            {Object.keys(data.clause_types || {}).length > 0 && (
              <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: '12px 14px' }}>
                <div style={{ ...MONO, fontSize: 10, color: 'var(--text3)', letterSpacing: '0.07em', marginBottom: 10 }}>CLAUSE TYPES DETECTED</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {Object.entries(data.clause_types).slice(0, 16).map(([type, count]) => (
                    <div key={type} style={{
                      display: 'flex', alignItems: 'center', gap: 5,
                      background: 'rgba(201,168,76,0.07)', border: '1px solid rgba(201,168,76,0.2)',
                      borderRadius: 20, padding: '3px 10px',
                    }}>
                      <span style={{ fontSize: 12, color: 'var(--text2)', textTransform: 'capitalize' }}>{type}</span>
                      <span style={{ ...MONO, fontSize: 9.5, color: 'var(--gold)', fontWeight: 700 }}>{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Scan status */}
            <div style={{ ...MONO, fontSize: 10.5, color: data.scan_done ? 'var(--green)' : data.scan_running ? 'var(--orange)' : 'var(--text3)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: data.scan_done ? 'var(--green)' : data.scan_running ? 'var(--orange)' : 'var(--border2)', animation: data.scan_running ? 'blink 0.8s infinite' : 'none' }} />
              {data.scan_done ? 'Full risk scan complete' : data.scan_running ? 'Risk scan in progress...' : 'No risk scan yet'}
            </div>

            {/* Quick action prompts */}
            <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: '12px 14px' }}>
              <div style={{ ...MONO, fontSize: 10, color: 'var(--text3)', letterSpacing: '0.07em', marginBottom: 10 }}>QUICK QUESTIONS</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {QUICK_QUESTIONS.map((q, i) => {
                  const c = INTENT_COLORS[q.intent]
                  return (
                    <button key={i}
                      onClick={() => { onAsk(q.text); onClose() }}
                      style={{
                        background: 'var(--bg2)', border: '1px solid var(--border)',
                        borderRadius: 8, padding: '9px 12px', cursor: 'pointer', textAlign: 'left',
                        display: 'flex', alignItems: 'center', gap: 8, transition: 'all 0.15s',
                      }}
                      onMouseEnter={e => { e.currentTarget.style.borderColor = c.border; e.currentTarget.style.background = c.bg }}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.background = 'var(--bg2)' }}>
                      <ChevronRight size={11} style={{ color: c.color, flexShrink: 0 }} />
                      <span style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.4 }}>{q.text}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          </>
        )}
      </div>
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
  const [activePanel, setActivePanel] = useState(null)  // 'risk' | 'analysis' | null
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

  // Send a question from the panel into the chatbox and close panel
  const explainRisk = async (text) => {
    setActivePanel(null)
    if (!text?.trim()) return
    setLoading(true)
    resetAgents()
    const uid = Date.now().toString()
    setMessages(prev => [...prev, { id: uid, role: 'user', content: text, timestamp: new Date() }])
    const lid = uid + '_loading'
    setMessages(prev => [...prev, { id: lid, role: 'assistant', loading: true, loadingText: 'Analysing risk...', timestamp: new Date() }])
    try {
      const res = await fetch(`${API}/ask`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: text, tone }) })
      const data = await res.json()
      const usedAgents = data.agents_used || []
      usedAgents.forEach(name => setAgentStates(prev => ({ ...prev, [name]: 'running' })))
      await new Promise(r => setTimeout(r, 600))
      usedAgents.forEach(name => setAgentStates(prev => ({ ...prev, [name]: 'done' })))
      setMessages(prev => prev.map(m => m.id === lid ? { id: lid, role: 'assistant', content: data.synthesis || data.error || 'No response.', response: data.success ? data : null, timestamp: new Date() } : m))
      if (data.success) { setSidebarTab('agents'); await fetchStatus() }
    } catch (e) {
      setMessages(prev => prev.map(m => m.id === lid ? { id: lid, role: 'assistant', content: `Connection error: ${e.message}`, timestamp: new Date() } : m))
    }
    setLoading(false)
  }

  const online = serverStatus?.status === 'online'

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="grid-bg" style={{ height: '100vh', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>

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
            { label: 'ANALYSIS', icon: Brain, panel: 'analysis' },
            { label: 'RISK', icon: Shield, panel: 'risk' },
          ].map(({ label, icon: Icon, panel }) => (
            <button
              key={label}
              onClick={() => setActivePanel(p => p === panel ? null : panel)}
              style={{
                display: 'flex', alignItems: 'center', gap: 6, padding: '5px 12px', borderRadius: 7,
                fontFamily: "'JetBrains Mono',monospace", fontSize: 10.5, letterSpacing: '0.05em',
                cursor: 'pointer', border: '1px solid',
                borderColor: activePanel === panel ? (panel === 'risk' ? 'rgba(255,128,64,0.5)' : 'rgba(0,212,255,0.5)') : 'transparent',
                color: activePanel === panel ? (panel === 'risk' ? 'var(--orange)' : 'var(--cyan)') : 'var(--text3)',
                background: activePanel === panel ? (panel === 'risk' ? 'rgba(255,128,64,0.08)' : 'rgba(0,212,255,0.08)') : 'transparent',
                transition: 'all 0.2s',
              }}>
              <Icon size={11} /> {label}
            </button>
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
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>

        {/* ── Left sidebar — fixed height, only inner content scrolls ── */}
        <aside style={{ width: 300, flexShrink: 0, borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg2)' }}>

          {/* Sidebar tabs — pinned at top, never scrolls */}
          <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', background: 'var(--bg)', flexShrink: 0 }}>
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
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0, minHeight: 0 }}>

          {/* Messages — only this scrolls */}
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
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

          {/* ── Right panel: Risk or Analysis ── */}
          {activePanel === 'risk' && (
            <RiskPanel
              onClose={() => setActivePanel(null)}
              onExplain={explainRisk}
            />
          )}
          {activePanel === 'analysis' && (
            <AnalysisPanel
              onClose={() => setActivePanel(null)}
              onAsk={explainRisk}
            />
          )}
          </div>  {/* end flex row wrapper */}

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