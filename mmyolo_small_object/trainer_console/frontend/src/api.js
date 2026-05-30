import axios from 'axios'

const api = axios.create({
  baseURL: '/',
  headers: {
    'Content-Type': 'application/json',
  },
})

export async function fetchJson(url, options = {}) {
  const method = options.method || 'get'
  const data = options.body ? JSON.parse(options.body) : undefined

  try {
    const response = await api.request({ url, method, data })
    return response.data
  } catch (error) {
    if (error.response) {
      const { status, statusText, data: body } = error.response
      let detail = `${status} ${statusText}`
      if (body && body.detail) {
        detail = body.detail
      } else if (typeof body === 'string' && body) {
        detail = body
      }
      throw new Error(detail)
    }
    throw error
  }
}

export const apiClient = {
  getHealth() {
    return fetchJson('/api/health')
  },
  getPresets() {
    return fetchJson('/api/presets')
  },
  getJobs() {
    return fetchJson('/api/jobs')
  },
  createJob(payload) {
    return fetchJson('/api/jobs', {
      method: 'post',
      body: JSON.stringify(payload),
    })
  },
  getJob(jobId) {
    return fetchJson(`/api/jobs/${jobId}`)
  },
  stopJob(jobId) {
    return fetchJson(`/api/jobs/${jobId}/stop`, { method: 'post' })
  },
  getJobLogs(jobId, maxBytes = 30000) {
    return fetchJson(`/api/jobs/${jobId}/logs?max_bytes=${maxBytes}`)
  },
  inspectDataset(path) {
    return fetchJson('/api/dataset/inspect', {
      method: 'post',
      body: JSON.stringify({ path }),
    })
  },
  scanDatasets() {
    return fetchJson('/api/datasets/scan')
  },
}
