// api.js
import axios from 'axios'

const BASE_URL = 'http://localhost:8000'

export const getMatches = async () => {
  const res = await axios.get(`${BASE_URL}/matches`)
  return res.data.matches
}

export const askQuestion = async (question, matchContext = null, history = []) => {
  const res = await axios.post(`${BASE_URL}/chat`, {
    question,
    match_context: matchContext,
    history,
  })
  return res.data
}