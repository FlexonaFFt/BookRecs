const USER_KEY = 'bookrecs_demo_user_id';

export function getStoredUserId() {
  try {
    return localStorage.getItem(USER_KEY) || '';
  } catch {
    return '';
  }
}

export function setStoredUserId(userId) {
  try {
    if (userId) {
      localStorage.setItem(USER_KEY, String(userId));
    }
  } catch {
    // no-op
  }
}

export async function fetchDemoUsers(limit = 200) {
  const res = await fetch(`/v1/demo/users?limit=${encodeURIComponent(limit)}`);
  if (!res.ok) {
    throw new Error(`users http ${res.status}`);
  }
  const payload = await res.json();
  return Array.isArray(payload.items) ? payload.items : [];
}

export async function fetchRecommendations(userId, topK = 10) {
  const res = await fetch('/v1/demo/recommendations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: String(userId), top_k: topK }),
  });
  if (!res.ok) {
    throw new Error(`recommendations http ${res.status}`);
  }
  return res.json();
}

export async function fetchDemoBook(itemId) {
  const res = await fetch(`/v1/demo/books/${encodeURIComponent(itemId)}`);
  if (!res.ok) {
    throw new Error(`book http ${res.status}`);
  }
  return res.json();
}

export async function fetchDemoBooksByIds(itemIds) {
  const unique = [...new Set(itemIds)].filter((x) => x !== null && x !== undefined);
  const results = await Promise.all(
    unique.map(async (id) => {
      try {
        const book = await fetchDemoBook(id);
        return [String(id), book];
      } catch {
        return [String(id), null];
      }
    })
  );
  return Object.fromEntries(results);
}

export async function fetchSimilarItems(itemId, limit = 8) {
  const res = await fetch(`/v1/items/${encodeURIComponent(itemId)}/similar?limit=${encodeURIComponent(limit)}`);
  if (!res.ok) {
    throw new Error(`similar http ${res.status}`);
  }
  return res.json();
}
