/**
 * HTTP client for the Avocado Ripeness API.
 *
 * By default requests go to the same origin under /api/v1:
 *  - in development, CRA's "proxy" (package.json) forwards them to localhost:8000;
 *  - in Docker, nginx forwards them to the API container.
 * Set REACT_APP_API_URL to call an API hosted elsewhere.
 */
const API_BASE_URL = `${(process.env.REACT_APP_API_URL || '').replace(/\/$/, '')}/api/v1`;

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);

  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {
      /* body was not JSON */
    }
    // FastAPI validation errors return a list; keep the message readable.
    const message = typeof detail === 'string' ? detail : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

/** Analyze one photo. Photos sharing a sessionId are fused as the same avocado. */
export function analyzeImage(file, sessionId) {
  const form = new FormData();
  form.append('file', file);
  form.append('session_id', sessionId);
  return request('/analyze', { method: 'POST', body: form });
}

/** @param {{session_id, image_id, predicted_stage, is_correct, correct_stage?}} feedback */
export function sendFeedback(feedback) {
  return request('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(feedback),
  });
}

/** Resolves to true when the API is up and the model is loaded. */
export async function checkHealth() {
  try {
    const health = await request('/health');
    return health.model_loaded;
  } catch {
    return false;
  }
}

/** Forget a session on the server (fire-and-forget; survives page unload). */
export function clearSession(sessionId) {
  return request(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
    keepalive: true,
  }).catch(() => {});
}
