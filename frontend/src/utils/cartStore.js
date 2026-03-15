const CART_EVENT = 'bookrecs-cart-updated';

function keyForUser(userId) {
  return `bookrecs_cart_${String(userId || 'guest')}`;
}

function safeParse(raw) {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function emitCartUpdate(userId) {
  try {
    window.dispatchEvent(new CustomEvent(CART_EVENT, { detail: { userId: String(userId || 'guest') } }));
  } catch {
    // no-op
  }
}

export function getCartItems(userId) {
  try {
    return safeParse(localStorage.getItem(keyForUser(userId)));
  } catch {
    return [];
  }
}

export function setCartItems(userId, items) {
  try {
    localStorage.setItem(keyForUser(userId), JSON.stringify(items || []));
  } catch {
    // no-op
  }
  emitCartUpdate(userId);
}

export function getCartCount(userId) {
  return getCartItems(userId).length;
}

export function addCartItem(userId, item) {
  const items = getCartItems(userId);
  const normalized = {
    item_id: Number(item?.item_id ?? 0),
    title: String(item?.title || 'Untitled Book'),
    partLabel: String(item?.partLabel || 'Standalone'),
    price: String(item?.price || '$0.00'),
  };
  items.push(normalized);
  setCartItems(userId, items);
}

export function removeCartItem(userId, idx) {
  const items = getCartItems(userId);
  const safeIdx = Number(idx);
  if (Number.isNaN(safeIdx) || safeIdx < 0 || safeIdx >= items.length) return;
  items.splice(safeIdx, 1);
  setCartItems(userId, items);
}

export function clearCart(userId) {
  setCartItems(userId, []);
}

export function onCartUpdate(handler) {
  const wrapped = (event) => handler(event?.detail?.userId || 'guest');
  window.addEventListener(CART_EVENT, wrapped);
  return () => window.removeEventListener(CART_EVENT, wrapped);
}
