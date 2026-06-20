// App.jsx
import { useState, useEffect, useRef } from 'react'
import { getMatches, askQuestion } from './api'
import ReactMarkdown from 'react-markdown'

export default function App() {
  const [matches, setMatches] = useState([])
  const [selectedMatch, setSelectedMatch] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingMatches, setLoadingMatches] = useState(true)
  const bottomRef = useRef(null)

  useEffect(() => {
    getMatches()
      .then(data => {
        setMatches(data)
        setSelectedMatch(data[0])
      })
      .catch(() => console.error('Failed to load matches'))
      .finally(() => setLoadingMatches(false))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleMatchSelect = (match) => {
    setSelectedMatch(match)
    setMessages([])
  }

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const question = input.trim()
    setInput('')

    setMessages(prev => [...prev, { role: 'user', content: question }])
    setLoading(true)

    try {
      const matchContext = selectedMatch
        ? `${selectedMatch.home_team} vs ${selectedMatch.away_team}`
        : null

      const history = messages.slice(-4).map(m => ({
        role: m.role,
        content: m.content,
      }))

      const data = await askQuestion(question, matchContext, history)
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, something went wrong. Make sure the backend is running.',
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div style={styles.container}>

      {/* SIDEBAR */}
      <div style={styles.sidebar}>
        <div style={styles.logo}>
          <span style={styles.logoIcon}></span>
          <span style={styles.logoText}>TacticLens</span>
        </div>

        <p style={styles.sidebarLabel}>SELECT MATCH</p>

        {loadingMatches ? (
          <p style={styles.loadingText}>Loading matches...</p>
        ) : (
          matches.map(match => (
            <div
              key={match.match_id}
              style={{
                ...styles.matchCard,
                ...(selectedMatch?.match_id === match.match_id
                  ? styles.matchCardActive
                  : {}),
              }}
              onClick={() => handleMatchSelect(match)}
            >
              <p style={styles.matchTeams}>
                {match.home_team} vs {match.away_team}
              </p>
              <p style={styles.matchScore}>
                {match.home_score} - {match.away_score}
              </p>
              <p style={styles.matchMeta}>
                {match.competition} · {match.match_date}
              </p>
            </div>
          ))
        )}

        <div style={styles.sidebarFooter}>
          <p style={styles.footerText}>Powered by</p>
          <p style={styles.footerStack}>Mistral · ChromaDB · LangChain</p>
        </div>
      </div>

      {/* MAIN CHAT AREA */}
      <div style={styles.main}>

        {/* Header */}
        <div style={styles.header}>
          {selectedMatch ? (
            <>
              <p style={styles.headerTitle}>
                {selectedMatch.home_team} vs {selectedMatch.away_team}
              </p>
              <p style={styles.headerMeta}>
                {selectedMatch.competition} · {selectedMatch.season} · {selectedMatch.match_date}
              </p>
            </>
          ) : (
            <p style={styles.headerTitle}>Select a match to begin</p>
          )}
        </div>

        {/* Messages */}
        <div style={styles.messages}>
          {messages.length === 0 && (
            <div style={styles.emptyState}>
              <p style={styles.emptyIcon}>📊</p>
              <p style={styles.emptyTitle}>Ask anything about this match</p>
              <p style={styles.emptySubtitle}>Try: "Who scored?" or "Which player made the most passes?"</p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              style={{
                ...styles.message,
                ...(msg.role === 'user' ? styles.userMessage : styles.aiMessage),
              }}
            >
              <span style={styles.messageRole}>
                {msg.role === 'user' ? 'You' : '⚽ TacticLens'}
              </span>
              <div style={styles.messageContent}>
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
            </div>
          ))}

          {loading && (
            <div style={{ ...styles.message, ...styles.aiMessage }}>
              <span style={styles.messageRole}>⚽ TacticLens</span>
              <p style={styles.typingIndicator}>Analyzing match data...</p>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={styles.inputArea}>
          <textarea
            style={styles.input}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about players, shots, passes, goals..."
            rows={1}
            disabled={loading}
          />
          <button
            style={{
              ...styles.sendButton,
              ...(loading ? styles.sendButtonDisabled : {}),
            }}
            onClick={handleSend}
            disabled={loading}
          >
            {loading ? '...' : '→'}
          </button>
        </div>
      </div>
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    height: '100vh',
    background: '#0f1117',
  },
  sidebar: {
    width: '280px',
    minWidth: '280px',
    background: '#1a1d27',
    borderRight: '1px solid #2d3148',
    display: 'flex',
    flexDirection: 'column',
    padding: '20px',
    gap: '8px',
    overflowY: 'auto',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '24px',
  },
  logoIcon: { fontSize: '24px' },
  logoText: {
    fontSize: '20px',
    fontWeight: '700',
    color: '#fff',
    letterSpacing: '-0.5px',
  },
  sidebarLabel: {
    fontSize: '11px',
    fontWeight: '600',
    color: '#4a5568',
    letterSpacing: '1px',
    marginBottom: '8px',
  },
  loadingText: { color: '#4a5568', fontSize: '14px' },
  matchCard: {
    padding: '14px',
    borderRadius: '10px',
    background: '#0f1117',
    border: '1px solid #2d3148',
    cursor: 'pointer',
    transition: 'all 0.2s',
    marginBottom: '6px',
  },
  matchCardActive: {
    border: '1px solid #4f6ef7',
    background: '#1a2040',
  },
  matchTeams: {
    fontSize: '13px',
    fontWeight: '600',
    color: '#e2e8f0',
    marginBottom: '4px',
  },
  matchScore: {
    fontSize: '22px',
    fontWeight: '700',
    color: '#4f6ef7',
    marginBottom: '4px',
  },
  matchMeta: {
    fontSize: '11px',
    color: '#4a5568',
  },
  sidebarFooter: {
    marginTop: 'auto',
    paddingTop: '20px',
    borderTop: '1px solid #2d3148',
  },
  footerText: { fontSize: '11px', color: '#4a5568' },
  footerStack: { fontSize: '12px', color: '#6b7280', marginTop: '4px' },
  main: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  header: {
    padding: '20px 28px',
    borderBottom: '1px solid #2d3148',
    background: '#1a1d27',
  },
  headerTitle: {
    fontSize: '18px',
    fontWeight: '700',
    color: '#fff',
  },
  headerMeta: {
    fontSize: '13px',
    color: '#4a5568',
    marginTop: '4px',
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: '24px 28px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  emptyState: {
    margin: 'auto',
    textAlign: 'center',
    padding: '40px',
  },
  emptyIcon: { fontSize: '48px', marginBottom: '16px' },
  emptyTitle: {
    fontSize: '18px',
    fontWeight: '600',
    color: '#e2e8f0',
    marginBottom: '8px',
  },
  emptySubtitle: {
    fontSize: '14px',
    color: '#4a5568',
    maxWidth: '360px',
    lineHeight: '1.6',
  },
  message: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    maxWidth: '75%',
  },
  userMessage: { alignSelf: 'flex-end', alignItems: 'flex-end' },
  aiMessage: { alignSelf: 'flex-start', alignItems: 'flex-start' },
  messageRole: {
    fontSize: '11px',
    fontWeight: '600',
    color: '#4a5568',
    letterSpacing: '0.5px',
  },
  messageContent: {
    padding: '14px 18px',
    borderRadius: '14px',
    fontSize: '14px',
    lineHeight: '1.7',
    background: '#1a1d27',
    border: '1px solid #2d3148',
    color: '#e2e8f0',
  },
  typingIndicator: {
    padding: '14px 18px',
    borderRadius: '14px',
    fontSize: '14px',
    color: '#4a5568',
    background: '#1a1d27',
    border: '1px solid #2d3148',
    fontStyle: 'italic',
  },
  inputArea: {
    padding: '16px 28px',
    borderTop: '1px solid #2d3148',
    display: 'flex',
    gap: '12px',
    alignItems: 'flex-end',
    background: '#1a1d27',
  },
  input: {
    flex: 1,
    padding: '14px 18px',
    borderRadius: '12px',
    border: '1px solid #2d3148',
    background: '#0f1117',
    color: '#e2e8f0',
    fontSize: '14px',
    resize: 'none',
    outline: 'none',
    fontFamily: 'inherit',
    lineHeight: '1.5',
  },
  sendButton: {
    width: '48px',
    height: '48px',
    borderRadius: '12px',
    background: '#4f6ef7',
    color: '#fff',
    border: 'none',
    fontSize: '20px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '700',
  },
  sendButtonDisabled: {
    background: '#2d3148',
    cursor: 'not-allowed',
  },
}
