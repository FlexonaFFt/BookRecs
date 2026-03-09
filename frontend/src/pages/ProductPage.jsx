import React, { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import UserSwitcher from '../components/UserSwitcher';
import {
  fetchDemoBook,
  fetchDemoBooksByIds,
  fetchDemoUsers,
  postInteraction,
  fetchRecommendations,
  fetchSimilarItems,
  getStoredUserId,
  setStoredUserId,
} from '../api/demoApi';
import { extractPartLabel, formatDisplayTitle, splitTitleForCover } from '../utils/bookFormat';
import { addCartItem, getCartCount, onCartUpdate } from '../utils/cartStore';

const fallbackBook = {
  item_id: 1,
  title: 'The Art of Silent Hours',
  description: 'Book details are unavailable right now.',
  tags: [],
  series: [],
};

function priceFromId(itemId) {
  const base = 20 + (Number(itemId) % 16);
  return `$${base}.00`;
}

function normalizeBook(raw) {
  const id = Number(raw?.item_id ?? 0);
  const rawTitle = raw?.title || `Book ${id || ''}`;
  const title = formatDisplayTitle(rawTitle);
  return {
    item_id: id,
    title,
    titleLines: splitTitleForCover(rawTitle),
    partLabel: extractPartLabel(raw?.title || ''),
    series: Array.isArray(raw?.series) && raw.series.length > 0 ? String(raw.series[0]) : 'Standalone',
    description: String(raw?.description || ''),
    tags: Array.isArray(raw?.tags) ? raw.tags : [],
    price: priceFromId(id),
    color: colorKeys[id % colorKeys.length],
  };
}

const colorKeys = ['green', 'burgundy', 'navy', 'terracotta', 'charcoal', 'olive', 'purple', 'blue'];

const catalogGradients = {
  green: ['linear-gradient(160deg, #3B5249 0%, #2A3D33 100%)', 'linear-gradient(90deg, #1E2B24 0%, #2A3D33 100%)'],
  burgundy: ['linear-gradient(160deg, #5D2E2E 0%, #3A1C1C 100%)', 'linear-gradient(90deg, #3A1C1C 0%, #5D2E2E 100%)'],
  navy: ['linear-gradient(160deg, #2E3B5D 0%, #1C243A 100%)', 'linear-gradient(90deg, #1C243A 0%, #2E3B5D 100%)'],
  terracotta: ['linear-gradient(160deg, #8C4B3E 0%, #5D322A 100%)', 'linear-gradient(90deg, #5D322A 0%, #8C4B3E 100%)'],
  charcoal: ['linear-gradient(160deg, #3D3D3D 0%, #262626 100%)', 'linear-gradient(90deg, #262626 0%, #3D3D3D 100%)'],
  olive: ['linear-gradient(160deg, #555D2E 0%, #363A1C 100%)', 'linear-gradient(90deg, #363A1C 0%, #555D2E 100%)'],
  purple: ['linear-gradient(160deg, #4A3B52 0%, #2F2236 100%)', 'linear-gradient(90deg, #2F2236 0%, #4A3B52 100%)'],
  blue: ['linear-gradient(160deg, #3B4A52 0%, #222F36 100%)', 'linear-gradient(90deg, #222F36 0%, #3B4A52 100%)'],
};

function CatalogLikeBookCard({ book, onAddToCart }) {
  const [hovered, setHovered] = useState(false);
  const [cover, spine] = catalogGradients[book.color] || catalogGradients.green;
  return (
    <article onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
      <Link to={`/book/${book.item_id}`} style={{ textDecoration: 'none' }}>
        <div style={{ perspective: '2000px', display: 'flex', justifyContent: 'center', alignItems: 'center', height: '280px', marginBottom: '12px' }}>
          <div
            style={{
              transformStyle: 'preserve-3d',
              transform: hovered ? 'scale(0.98) rotateY(-10deg) rotateX(5deg)' : 'scale(0.88) rotateY(-30deg) rotateX(10deg)',
              transition: 'transform 0.7s cubic-bezier(0.25,0.46,0.45,0.94)',
            }}
          >
            <div style={{ position: 'relative', width: '180px', height: '252px', transformStyle: 'preserve-3d' }}>
              <div
                style={{
                  position: 'absolute',
                  width: '180px',
                  height: '252px',
                  transform: 'translateZ(22px)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '18px',
                  background: cover,
                  boxShadow: 'inset -4px 0 12px rgba(0,0,0,0.25)',
                }}
              >
                <div
                  style={{
                    width: '28px',
                    height: '28px',
                    borderRadius: '50%',
                    border: '1px solid #D4C89A',
                    opacity: '0.4',
                    position: 'absolute',
                    top: '14px',
                    right: '14px',
                  }}
                ></div>
                <div style={{ fontFamily: "'Cinzel', serif", color: '#D4C89A', fontSize: '14px', textAlign: 'center', textTransform: 'uppercase' }}>
                  {book.titleLines[0]}<br />{book.titleLines[1]}
                </div>
                <div style={{ width: '40px', height: '1px', background: '#D4C89A', margin: '12px 0', opacity: 0.6 }}></div>
                <div style={{ color: '#D4C89A', fontSize: '8px', letterSpacing: '0.15em', textTransform: 'uppercase', opacity: 0.8 }}>{book.partLabel}</div>
              </div>
              <div style={{ position: 'absolute', width: '46px', height: '252px', transform: 'rotateY(-90deg) translateZ(22px)', display: 'flex', alignItems: 'center', justifyContent: 'center', background: spine }}>
                <div style={{ fontFamily: "'Cinzel', serif", color: '#D4C89A', fontSize: '8px', letterSpacing: '0.12em', textTransform: 'uppercase', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>{book.title}</div>
              </div>
              <div style={{ position: 'absolute', width: '180px', height: '252px', background: '#222', transform: 'translateZ(-22px) rotateY(180deg)' }}></div>
              <div style={{ position: 'absolute', width: '44px', height: '248px', background: 'repeating-linear-gradient(to bottom, #F6F4EC 0px, #F6F4EC 1px, #E8E5DB 2px, #E8E5DB 3px)', transform: 'rotateY(90deg) translateZ(156px)', top: '2px' }}></div>
              <div style={{ position: 'absolute', width: '180px', height: '252px', transform: 'translateZ(-23px)', boxShadow: hovered ? '0 24px 42px rgba(0,0,0,0.25)' : '0 18px 30px rgba(0,0,0,0.18)' }}></div>
            </div>
          </div>
        </div>
      </Link>
      <div style={{ textAlign: 'center' }}>
        <h3 style={{ fontFamily: "'Cinzel', serif", marginBottom: '4px', fontSize: '28px' }}>
          <Link to={`/book/${book.item_id}`} style={{ textDecoration: 'none' }}>{book.title}</Link>
        </h3>
        <p style={{ fontSize: '10px', textTransform: 'uppercase', color: '#6b7280', marginBottom: '12px' }}>{book.partLabel}</p>
        <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #ddd', paddingTop: '10px' }}>
          <span style={{ fontWeight: 500, fontSize: '14px' }}>{book.price}</span>
          <button
            onClick={() => onAddToCart(book)}
            style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', background: 'none', border: 'none', borderBottom: '1px solid #1A1A1A', cursor: 'pointer' }}
          >
            Add to cart
          </button>
        </div>
      </div>
    </article >
  );
}

const previewPalettes = {
  green: {
    cover: 'linear-gradient(160deg, #3B5249 0%, #2A3D33 100%)',
    spine: 'linear-gradient(90deg, #1E2B24 0%, #2A3D33 100%)',
    back: '#1E2B24',
  },
  burgundy: {
    cover: 'linear-gradient(160deg, #5D2E2E 0%, #3A1C1C 100%)',
    spine: 'linear-gradient(90deg, #3A1C1C 0%, #5D2E2E 100%)',
    back: '#2B1212',
  },
  navy: {
    cover: 'linear-gradient(160deg, #2E3B5D 0%, #1C243A 100%)',
    spine: 'linear-gradient(90deg, #1C243A 0%, #2E3B5D 100%)',
    back: '#182033',
  },
  terracotta: {
    cover: 'linear-gradient(160deg, #8C4B3E 0%, #5D322A 100%)',
    spine: 'linear-gradient(90deg, #5D322A 0%, #8C4B3E 100%)',
    back: '#4D2A23',
  },
  charcoal: {
    cover: 'linear-gradient(160deg, #3D3D3D 0%, #262626 100%)',
    spine: 'linear-gradient(90deg, #262626 0%, #3D3D3D 100%)',
    back: '#1F1F1F',
  },
  olive: {
    cover: 'linear-gradient(160deg, #555D2E 0%, #363A1C 100%)',
    spine: 'linear-gradient(90deg, #363A1C 0%, #555D2E 100%)',
    back: '#2E3118',
  },
  purple: {
    cover: 'linear-gradient(160deg, #4A3B52 0%, #2F2236 100%)',
    spine: 'linear-gradient(90deg, #2F2236 0%, #4A3B52 100%)',
    back: '#281C2E',
  },
  blue: {
    cover: 'linear-gradient(160deg, #3B4A52 0%, #222F36 100%)',
    spine: 'linear-gradient(90deg, #222F36 0%, #3B4A52 100%)',
    back: '#1D2830',
  },
};

const previewPaletteKeys = Object.keys(previewPalettes);

function Book3DPreview({ title, partLabel, palette, hovered }) {
  const parts = splitTitleForCover(title || '');
  const displayTitle = formatDisplayTitle(title || '');
  const colors = previewPalettes[palette] || previewPalettes.green;
  return (
    <div style={{ perspective: '1600px', display: 'flex', justifyContent: 'center' }}>
      <div
        style={{
          width: '220px',
          height: '310px',
          position: 'relative',
          transformStyle: 'preserve-3d',
          transform: hovered ? 'scale(1) rotateY(-15deg) rotateX(6deg)' : 'scale(0.98) rotateY(-25deg) rotateX(10deg)',
          transition: 'transform 0.5s ease',
        }}
      >
        <div
          style={{
            position: 'absolute',
            width: '220px',
            height: '310px',
            transform: 'translateZ(30px)',
            background: colors.cover,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '24px',
            boxShadow: 'inset -3px 0 8px rgba(0,0,0,0.25)',
          }}
        >
          <div style={{ fontFamily: "'Cinzel', serif", color: '#D4C89A', fontSize: '18px', textAlign: 'center', textTransform: 'uppercase', lineHeight: 1.35 }}>
            {parts[0]}<br />{parts[1]}
          </div>
          <div style={{ width: '58px', height: '1px', background: '#D4C89A', opacity: 0.6, margin: '14px 0' }}></div>
          <div style={{ fontSize: '10px', color: '#D4C89A', letterSpacing: '0.16em', textTransform: 'uppercase', opacity: 0.85 }}>{partLabel}</div>
        </div>
        <div
          style={{
            position: 'absolute',
            width: '60px',
            height: '310px',
            transform: 'rotateY(-90deg) translateZ(30px)',
            background: colors.spine,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden',
          }}
        >
          <div style={{ fontFamily: "'Cinzel', serif", color: '#D4C89A', fontSize: '11px', letterSpacing: '0.12em', textTransform: 'uppercase', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>{displayTitle}</div>
        </div>
        <div style={{ position: 'absolute', width: '220px', height: '310px', transform: 'translateZ(-30px) rotateY(180deg)', background: colors.back }}></div>
        <div style={{ position: 'absolute', width: '58px', height: '308px', transform: 'rotateY(90deg) translateZ(189px)', top: '1px', background: 'repeating-linear-gradient(to bottom, #F6F4EC 0px, #F6F4EC 1px, #E8E5DB 2px, #E8E5DB 3px)' }}></div>
        <div style={{ position: 'absolute', width: '220px', height: '310px', transform: 'translateZ(-31px)', boxShadow: hovered ? '0 26px 36px rgba(0,0,0,0.26)' : '0 18px 28px rgba(0,0,0,0.2)' }}></div>
      </div>
    </div>
  );
}

export default function ProductPage() {
  const { itemId } = useParams();

  const [book, setBook] = useState(fallbackBook);
  const [loadingBook, setLoadingBook] = useState(false);

  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState('');
  const [cartCount, setCartCount] = useState(0);
  const [eventStatus, setEventStatus] = useState('');
  const [previewHovered, setPreviewHovered] = useState(false);

  const [similarBooks, setSimilarBooks] = useState([]);
  const [recommendedBooks, setRecommendedBooks] = useState([]);

  useEffect(() => {
    let cancelled = false;
    async function loadUsers() {
      try {
        const list = await fetchDemoUsers(300);
        if (!cancelled) {
          setUsers(list);
          const stored = getStoredUserId();
          const available = new Set(list.map((x) => x.user_id));
          const initial = stored && available.has(stored) ? stored : (list[0]?.user_id || '');
          setSelectedUserId(initial);
          if (initial) {
            setStoredUserId(initial);
          }
        }
      } catch {
        if (!cancelled) {
          setUsers([]);
        }
      }
    }
    loadUsers();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setCartCount(getCartCount(selectedUserId));
    const off = onCartUpdate((updatedUserId) => {
      if (!selectedUserId || !updatedUserId || updatedUserId === String(selectedUserId)) {
        setCartCount(getCartCount(selectedUserId));
      }
    });
    return off;
  }, [selectedUserId]);

  useEffect(() => {
    let cancelled = false;
    async function loadBook() {
      if (!itemId) {
        return;
      }
      setLoadingBook(true);
      try {
        const raw = await fetchDemoBook(itemId);
        if (!cancelled) {
          setBook(normalizeBook(raw));
        }
      } catch {
        if (!cancelled) {
          setBook(normalizeBook({ ...fallbackBook, item_id: Number(itemId || 1) }));
        }
      } finally {
        if (!cancelled) {
          setLoadingBook(false);
        }
      }
    }
    loadBook();
    return () => {
      cancelled = true;
    };
  }, [itemId]);

  useEffect(() => {
    let cancelled = false;
    async function loadSimilar() {
      if (!itemId) {
        return;
      }
      try {
        const payload = await fetchSimilarItems(itemId, 8);
        const ids = [
          ...(payload?.content || []).map((x) => x.item_id),
          ...(payload?.cf || []).map((x) => x.item_id),
        ].filter((x) => String(x) !== String(itemId));
        const unique = [...new Set(ids)].slice(0, 8);
        const map = await fetchDemoBooksByIds(unique);
        const items = unique
          .map((id) => map[String(id)])
          .filter(Boolean)
          .map((raw) => normalizeBook(raw));
        if (!cancelled) {
          setSimilarBooks(items);
        }
      } catch {
        if (!cancelled) {
          setSimilarBooks([]);
        }
      }
    }
    loadSimilar();
    return () => {
      cancelled = true;
    };
  }, [itemId]);

  useEffect(() => {
    let cancelled = false;
    async function loadRecs() {
      if (!selectedUserId) {
        return;
      }
      try {
        const reco = await fetchRecommendations(selectedUserId, 4);
        const ids = (reco?.items || []).map((x) => x.item_id).filter((x) => String(x) !== String(itemId));
        const unique = [...new Set(ids)].slice(0, 4);
        const map = await fetchDemoBooksByIds(unique);
        const items = unique
          .map((id) => map[String(id)])
          .filter(Boolean)
          .map((raw) => normalizeBook(raw));
        if (!cancelled) {
          setRecommendedBooks(items);
        }
      } catch {
        if (!cancelled) {
          setRecommendedBooks([]);
        }
      }
    }
    loadRecs();
    return () => {
      cancelled = true;
    };
  }, [selectedUserId, itemId]);

  const price = useMemo(() => priceFromId(book.item_id), [book.item_id]);
  const shortDescription = useMemo(() => {
    const text = String(book.description || '');
    if (!text) {
      return 'Description is unavailable for this title in the current dataset.';
    }
    return text.length > 220 ? `${text.slice(0, 220).trim()}...` : text;
  }, [book.description]);
  const previewPalette = previewPaletteKeys[Math.abs(Number(book.item_id || 0)) % previewPaletteKeys.length];
  const onAddToCart = async (bookArg = book) => {
    addCartItem(selectedUserId, {
      item_id: bookArg.item_id,
      title: bookArg.title,
      partLabel: bookArg.partLabel,
      price: bookArg.price || priceFromId(bookArg.item_id),
    });
    try {
      await postInteraction({
        userId: selectedUserId,
        itemId: bookArg.item_id,
        eventType: 'add_to_cart',
      });
      setEventStatus('Added to cart');
    } catch {
      setEventStatus('Added to cart (offline)');
    }
    setTimeout(() => setEventStatus(''), 1800);
  };
  const onSimulatePurchase = async () => {
    try {
      await postInteraction({
        userId: selectedUserId,
        itemId: book.item_id,
        eventType: 'purchase',
      });
      setEventStatus('Purchase simulated');
    } catch {
      setEventStatus('Purchase simulated (offline)');
    }
    setTimeout(() => setEventStatus(''), 1800);
  };

  return (
    <div style={{ backgroundColor: '#EAE8E0', minHeight: '100vh', padding: '20px' }}>
      <div style={{ backgroundColor: '#F6F4EC', border: '1px solid #1A1A1A', maxWidth: '1400px', margin: '0 auto' }}>
        <header style={{ height: '60px', borderBottom: '1px solid #1A1A1A', display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', padding: '0 24px' }}>
          <nav style={{ display: 'flex', gap: '24px', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            <Link to="/catalog" style={{ textDecoration: 'none' }}>Catalog</Link>
            <Link to="/" style={{ textDecoration: 'none' }}>Home</Link>
          </nav>
          <div style={{ fontFamily: "'Cinzel', serif", fontSize: '24px' }}>Folio.</div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '18px', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            <UserSwitcher
              users={users}
              selectedUserId={selectedUserId}
              onChange={(userId) => {
                setSelectedUserId(userId);
                setStoredUserId(userId);
              }}
            />
            <Link to="/cart" style={{ textDecoration: 'none' }}>Cart ({cartCount})</Link>
          </div>
        </header>

        <section style={{ padding: '42px 24px 36px', borderBottom: '1px solid #1A1A1A' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', alignItems: 'center' }}>
            <div style={{ maxWidth: '500px', marginLeft: '34px' }}>
              <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#666', marginBottom: '10px' }}>Product Page</div>
              <h1 style={{ fontFamily: "'Cinzel', serif", fontSize: '34px', lineHeight: 1.1, marginBottom: '12px', maxWidth: '440px' }}>
                {loadingBook ? 'Loading...' : book.title}
              </h1>
              <p style={{ fontSize: '11px', textTransform: 'uppercase', color: '#6b7280', marginBottom: '12px' }}>{book.partLabel} · {book.series}</p>
              <p style={{ fontSize: '14px', lineHeight: 1.55, color: '#333', maxWidth: '440px', marginBottom: '16px' }}>
                {shortDescription}
              </p>
              <div style={{ fontSize: '10px', textTransform: 'uppercase', color: '#666', marginBottom: '6px' }}>Price</div>
              <div style={{ fontSize: '30px', fontWeight: 500, marginBottom: '12px' }}>{price}</div>
              <button
                onClick={onAddToCart}
                style={{ background: '#1A1A1A', color: '#F6F4EC', border: 'none', padding: '12px 18px', cursor: 'pointer', textTransform: 'uppercase', letterSpacing: '0.08em', fontSize: '11px' }}
              >
                Add to cart
              </button>
              <button
                onClick={onSimulatePurchase}
                style={{ marginLeft: '10px', background: 'transparent', color: '#1A1A1A', border: '1px solid #1A1A1A', padding: '11px 18px', cursor: 'pointer', textTransform: 'uppercase', letterSpacing: '0.08em', fontSize: '11px' }}
              >
                Simulate purchase
              </button>
              {eventStatus && (
                <div style={{ marginTop: '10px', fontSize: '11px', textTransform: 'uppercase', color: '#334155' }}>
                  {eventStatus}
                </div>
              )}
            </div>
            <div
              style={{ minHeight: '360px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}
              onMouseEnter={() => setPreviewHovered(true)}
              onMouseLeave={() => setPreviewHovered(false)}
            >
              <Book3DPreview title={book.title} partLabel={book.partLabel} palette={previewPalette} hovered={previewHovered} />
            </div>
          </div>
        </section>

        <section style={{ padding: '34px 24px 28px', borderBottom: '1px solid #1A1A1A' }}>
          <h2 style={{ fontFamily: "'Cinzel', serif", fontSize: '24px', marginBottom: '20px' }}>Recommended For This User</h2>
          {recommendedBooks.length === 0 ? (
            <div style={{ fontSize: '12px', color: '#666' }}>No personalized recommendations available.</div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(220px, 1fr))', gap: '32px 24px' }}>
              {recommendedBooks.map((x) => <CatalogLikeBookCard key={`rec-${x.item_id}`} book={x} onAddToCart={onAddToCart} />)}
            </div>
          )}
        </section>

        <section style={{ padding: '34px 24px 28px' }}>
          <h2 style={{ fontFamily: "'Cinzel', serif", fontSize: '24px', marginBottom: '20px' }}>Similar Books</h2>
          {similarBooks.length === 0 ? (
            <div style={{ fontSize: '12px', color: '#666' }}>No similar books were found for this item.</div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(220px, 1fr))', gap: '32px 24px' }}>
              {similarBooks.map((x) => <CatalogLikeBookCard key={`sim-${x.item_id}`} book={x} onAddToCart={onAddToCart} />)}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
