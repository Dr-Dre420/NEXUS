// Single source of truth for API access. No component hardcodes a base URL.
const BASE = (import.meta.env?.VITE_API_BASE ?? 'http://127.0.0.1:8000').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, { status = null, detail = null, cause = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.cause = cause;
  }
}

async function parseFailure(res) {
  // The API returns JSON {detail} for handled errors and plain text for unhandled ones.
  const text = await res.text().catch(() => '');
  try {
    const j = JSON.parse(text);
    return j.detail || j.message || text;
  } catch {
    return text || res.statusText;
  }
}

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch (e) {
    throw new ApiError(
      `Cannot reach the NEXUS API at ${BASE}. Start it with: uvicorn src.api.main:app`,
      { cause: e },
    );
  }
  if (!res.ok) {
    const detail = await parseFailure(res);
    if (res.status === 503) {
      throw new ApiError('The analytical data store is still loading. Retry in a moment.', { status: 503, detail });
    }
    throw new ApiError(detail || `Request failed (${res.status})`, { status: res.status, detail });
  }
  try {
    return await res.json();
  } catch (e) {
    throw new ApiError('The API returned a malformed response.', { status: res.status, cause: e });
  }
}

const get = (p) => request(p, { method: 'GET' });
const post = (p, body) =>
  request(p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

// Immutable frozen-snapshot reads are cached for the session.
const cache = new Map();
const cached = (key, fn) => {
  if (!cache.has(key)) cache.set(key, fn().catch((e) => { cache.delete(key); throw e; }));
  return cache.get(key);
};

export const api = {
  base: BASE,
  health: () => get('/health'),
  portfolioSummary: () => cached('portfolio', () => get('/portfolio/summary')),
  borrower: (id) => cached(`b:${id}`, () => get(`/borrower/${encodeURIComponent(id)}`)),
  group: (id) => cached(`g:${id}`, () => get(`/group/${encodeURIComponent(id)}`)),
  network: () => cached('network', () => get('/network')),
  evaluation: () => cached('evaluation', () => get('/evaluation')),
  assumptions: () => cached('assumptions', () => get('/assumptions')),
  simulate: (body) => post('/simulate', body),
  intervene: (body) => post('/intervene', body),
};

/** Canonical identifier helpers. The API accepts "B10" or "10"; the UI always shows "B10". */
export const canonicalBorrower = (raw) => {
  const t = String(raw ?? '').trim().toUpperCase();
  if (!t) return null;
  if (/^B\d+$/.test(t)) return t;
  if (/^\d+$/.test(t)) return `B${t}`;
  return null;
};
export const canonicalGroup = (raw) => {
  const t = String(raw ?? '').trim().toUpperCase();
  if (!t) return null;
  if (/^G\d+$/.test(t)) return t;
  if (/^\d+$/.test(t)) return `G${t}`;
  return null;
};
