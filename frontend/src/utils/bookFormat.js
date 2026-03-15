export function formatDisplayTitle(rawTitle) {
  const source = String(rawTitle || '').trim();
  if (!source) {
    return 'Untitled Book';
  }

  let title = source;
  const bracket = source.match(/^(.*?)\s*\((.*?)\)\s*$/);
  if (bracket) {
    title = (bracket[1] || '').trim() || source;
  }

  title = title
    .replace(/[\s,:;-]*\bpart\s+\d+(?:\.\d+)?\b\s*$/i, '')
    .replace(/[\s,:;-]*#\s*\d+(?:\.\d+)?\s*$/i, '')
    .trim();

  return title || 'Untitled Book';
}

export function splitTitleForCover(rawTitle) {
  const title = formatDisplayTitle(rawTitle);
  const parts = title.split(/\s+/).filter(Boolean);
  if (parts.length <= 1) {
    return [title, ''];
  }
  const mid = Math.ceil(parts.length / 2);
  return [parts.slice(0, mid).join(' '), parts.slice(mid).join(' ')];
}

export function displayAuthor(authors) {
  if (Array.isArray(authors) && authors.length > 0 && String(authors[0]).trim()) {
    return String(authors[0]).trim();
  }
  return 'Author unavailable';
}

export function extractPartLabel(rawTitle) {
  const text = String(rawTitle || '').trim();
  if (!text) {
    return 'Standalone';
  }
  const hashPart = text.match(/#\s*(\d+(?:\.\d+)?)/i);
  if (hashPart) {
    return `Part ${hashPart[1]}`;
  }
  const explicitPart = text.match(/\bpart\s+(\d+(?:\.\d+)?)\b/i);
  if (explicitPart) {
    return `Part ${explicitPart[1]}`;
  }
  return 'Standalone';
}
