// api.js
import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || ''

export const getMatches = async () => {
  const res = await axios.get(`${BASE_URL}/matches`)
  return res.data.matches
}

export const askQuestion = async (question, matchContext = null, history = [], matchId = null) => {
  const res = await axios.post(`${BASE_URL}/chat`, {
    question,
    match_context: matchContext,
    match_id: matchId ? String(matchId) : null,
    history,
  })
  return res.data
}