import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import UserSwitcher from '../components/UserSwitcher';
import { extractPartLabel, formatDisplayTitle, splitTitleForCover } from '../utils/bookFormat';
import {
  fetchDemoBooksByIds,
  fetchDemoUsers,
  fetchRecommendations,
  getStoredUserId,
  setStoredUserId,
} from '../api/demoApi';

const styles = {
  body: { backgroundColor: '#EAE8E0', display: 'flex', justifyContent: 'center', minHeight: '100vh', padding: '20px' },
  marketContainer: { backgroundColor: '#F6F4EC', width: '100%', maxWidth: '1400px', border: '1px solid #1A1A1A', display: 'flex', flexDirection: 'column' },
  grid: { flex: 1, padding: '48px', display: 'grid', gridTemplateColumns: 'repeat(4, minmax(220px, 1fr))', gap: '32px 32px', rowGap: '64px', backgroundColor: '#F6F4EC' },
  bookWrapper: { perspective: '2000px', display: 'flex', justifyContent: 'center', alignItems: 'center', height: '380px', marginBottom: '18px' },
};

const fallbackBooks = [
  { id: 1, item_id: 1, title: 'The Art of Silent Hours', titleLines: ['The Art of', 'Silent Hours'], author: 'Eleanor Voss', price: '$24.00', color: 'green', genre: 'Fiction' },
  { id: 2, item_id: 2, title: 'Echoes of The Empire', titleLines: ['Echoes of', 'The Empire'], author: 'Marcus Harl', price: '$28.00', color: 'burgundy', genre: 'History' },
  { id: 3, item_id: 3, title: 'Deep Water Navigation', titleLines: ['Deep Water', 'Navigation'], author: 'S. J. Thorne', price: '$32.00', color: 'navy', genre: 'Science' },
  { id: 4, item_id: 4, title: 'Clay & Memory', titleLines: ['Clay &', 'Memory'], author: 'Ada L. Rose', price: '$22.00', color: 'terracotta', genre: 'Poetry' },
];

const colorKeys = ['green', 'burgundy', 'navy', 'terracotta', 'charcoal', 'olive', 'purple', 'blue'];

const gradients = {
  green: ['linear-gradient(160deg, #3B5249 0%, #2A3D33 100%)', 'linear-gradient(90deg, #1E2B24 0%, #2A3D33 100%)'],
  burgundy: ['linear-gradient(160deg, #5D2E2E 0%, #3A1C1C 100%)', 'linear-gradient(90deg, #3A1C1C 0%, #5D2E2E 100%)'],
  navy: ['linear-gradient(160deg, #2E3B5D 0%, #1C243A 100%)', 'linear-gradient(90deg, #1C243A 0%, #2E3B5D 100%)'],
  terracotta: ['linear-gradient(160deg, #8C4B3E 0%, #5D322A 100%)', 'linear-gradient(90deg, #5D322A 0%, #8C4B3E 100%)'],
  charcoal: ['linear-gradient(160deg, #3D3D3D 0%, #262626 100%)', 'linear-gradient(90deg, #262626 0%, #3D3D3D 100%)'],
  olive: ['linear-gradient(160deg, #555D2E 0%, #363A1C 100%)', 'linear-gradient(90deg, #363A1C 0%, #555D2E 100%)'],
  purple: ['linear-gradient(160deg, #4A3B52 0%, #2F2236 100%)', 'linear-gradient(90deg, #2F2236 0%, #4A3B52 100%)'],
  blue: ['linear-gradient(160deg, #3B4A52 0%, #222F36 100%)', 'linear-gradient(90deg, #222F36 0%, #3B4A52 100%)'],
};

function priceFromId(itemId) {
  const base = 20 + (Number(itemId) % 16);
  return `$${base}.00`;
}

function titleLines(title) {
  return splitTitleForCover(title);
}

function normalizeBook(raw, idx = 0) {
  const id = Number(raw.item_id ?? raw.id ?? idx + 1);
  const tags = Array.isArray(raw.tags) ? raw.tags : [];
  const rawTitle = raw.title || `Book ${id}`;
  return {
    id,
    item_id: id,
    title: formatDisplayTitle(rawTitle),
    titleLines: titleLines(rawTitle),
    partLabel: extractPartLabel(rawTitle),
    price: priceFromId(id),
    color: colorKeys[id % colorKeys.length],
    genre: tags.length > 0 ? String(tags[0]) : 'All',
  };
}

function BookCard({ book, onAddToCart }) {
  const [hovered, setHovered] = useState(false);
  const [cover, spine] = gradients[book.color];

  return (
    <article onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
      <Link to={`/book/${book.item_id}`} style={{ textDecoration: 'none' }}>
      <div style={styles.bookWrapper}>
        <div
          style={{
            transformStyle: 'preserve-3d',
            transform: hovered ? 'scale(1) rotateY(-10deg) rotateX(5deg)' : 'scale(0.9) rotateY(-32deg) rotateX(10deg)',
            transition: 'transform 0.7s cubic-bezier(0.25,0.46,0.45,0.94)',
          }}
        >
          <div style={{ position: 'relative', width: '200px', height: '280px', transformStyle: 'preserve-3d' }}>
            <div
              style={{
                position: 'absolute',
                width: '200px',
                height: '280px',
                transform: 'translateZ(25px)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '20px',
                background: cover,
                boxShadow: 'inset -4px 0 12px rgba(0,0,0,0.25)',
              }}
            >
              <div
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  border: '1px solid #D4C89A',
                  opacity: '0.4',
                  position: 'absolute',
                  top: '16px',
                  right: '16px',
                }}
              ></div>
              <div style={{ fontFamily: "'Cinzel', serif", color: '#D4C89A', fontSize: '16px', textAlign: 'center', textTransform: 'uppercase' }}>
                {book.titleLines[0]}<br />{book.titleLines[1]}
              </div>
              <div style={{ width: '40px', height: '1px', background: '#D4C89A', margin: '12px 0', opacity: 0.6 }}></div>
              <div style={{ color: '#D4C89A', fontSize: '8px', letterSpacing: '0.15em', textTransform: 'uppercase', opacity: 0.8 }}>{book.partLabel}</div>
            </div>

            <div style={{ position: 'absolute', width: '50px', height: '280px', transform: 'rotateY(-90deg) translateZ(25px)', display: 'flex', alignItems: 'center', justifyContent: 'center', background: spine }}>
              <div style={{ fontFamily: "'Cinzel', serif", color: '#D4C89A', fontSize: '9px', letterSpacing: '0.12em', textTransform: 'uppercase', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>{book.title}</div>
            </div>

            <div style={{ position: 'absolute', width: '200px', height: '280px', background: '#222', transform: 'translateZ(-25px) rotateY(180deg)' }}></div>
            <div style={{ position: 'absolute', width: '48px', height: '276px', background: 'repeating-linear-gradient(to bottom, #F6F4EC 0px, #F6F4EC 1px, #E8E5DB 2px, #E8E5DB 3px)', transform: 'rotateY(90deg) translateZ(174px)', top: '2px' }}></div>
            <div style={{ position: 'absolute', width: '200px', height: '280px', transform: 'translateZ(-26px)', boxShadow: hovered ? '0 24px 42px rgba(0,0,0,0.25)' : '0 18px 30px rgba(0,0,0,0.18)' }}></div>
          </div>
        </div>
      </div>
      </Link>

      <div style={{ textAlign: 'center' }}>
        <h3 style={{ fontFamily: "'Cinzel', serif", marginBottom: '4px' }}>
          <Link to={`/book/${book.item_id}`} style={{ textDecoration: 'none' }}>{book.title}</Link>
        </h3>
        <p style={{ fontSize: '10px', textTransform: 'uppercase', color: '#6b7280', marginBottom: '12px' }}>{book.partLabel}</p>
        <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #ddd', paddingTop: '10px' }}>
          <span style={{ fontWeight: 500, fontSize: '14px' }}>{book.price}</span>
          <button onClick={() => onAddToCart(book)} style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', background: 'none', border: 'none', borderBottom: '1px solid #1A1A1A', cursor: 'pointer' }}>Add to cart</button>
        </div>
      </div>
    </article>
  );
}

export default function CatalogPage() {
  const [activeFilter, setActiveFilter] = useState('All');
  const [cartCount, setCartCount] = useState(2);
  const [notification, setNotification] = useState(null);
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const [books, setBooks] = useState(fallbackBooks);
  const [total, setTotal] = useState(fallbackBooks.length);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState('');
  const [topRecs, setTopRecs] = useState([]);
  const [loadingRecs, setLoadingRecs] = useState(false);

  const limit = 24;
  const filters = ['All', 'Fiction', 'Poetry', 'Essays', 'History', 'Science'];

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
          setSelectedUserId('');
        }
      }
    }
    loadUsers();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function fetchCatalog() {
      setLoading(true);
      setError('');
      try {
        const offset = page * limit;
        const params = new URLSearchParams({
          limit: String(limit),
          offset: String(offset),
        });
        if (search.trim()) {
          params.set('q', search.trim());
        }
        if (activeFilter !== 'All') {
          params.set('genre', activeFilter.toLowerCase());
        }
        const res = await fetch(`/v1/demo/catalog?${params.toString()}`);
        if (!res.ok) {
          throw new Error(`catalog http ${res.status}`);
        }
        const payload = await res.json();
        const items = Array.isArray(payload.items)
          ? payload.items.map((x, idx) => normalizeBook(x, idx))
          : [];
        if (!cancelled) {
          setBooks(items);
          setTotal(Number(payload.total || 0));
        }
      } catch {
        if (!cancelled) {
          setError('Catalog API unavailable, fallback mode enabled');
          const filtered = fallbackBooks.filter((b) => {
            const genreOk = activeFilter === 'All' ? true : b.genre === activeFilter;
            const q = search.trim().toLowerCase();
            const qOk = q ? b.title.toLowerCase().includes(q) : true;
            return genreOk && qOk;
          });
          setBooks(filtered);
          setTotal(filtered.length);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchCatalog();
    return () => {
      cancelled = true;
    };
  }, [activeFilter, search, page]);

  useEffect(() => {
    let cancelled = false;
    async function loadTop10() {
      if (!selectedUserId) {
        setTopRecs([]);
        return;
      }
      setLoadingRecs(true);
      try {
        const reco = await fetchRecommendations(selectedUserId, 8);
        const items = Array.isArray(reco.items) ? reco.items : [];
        const map = await fetchDemoBooksByIds(items.map((x) => x.item_id));
        const enriched = items.map((x, idx) => {
          const raw = map[String(x.item_id)] || { item_id: x.item_id, title: `Book ${x.item_id}`, authors: [], tags: [] };
          return normalizeBook(raw, idx);
        });
        if (!cancelled) {
          setTopRecs(enriched);
        }
      } catch {
        if (!cancelled) {
          setTopRecs([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingRecs(false);
        }
      }
    }
    loadTop10();
    return () => {
      cancelled = true;
    };
  }, [selectedUserId]);

  const totalPages = useMemo(() => Math.max(1, Math.ceil(total / limit)), [total]);

  const onAddToCart = (book) => {
    setCartCount((prev) => prev + 1);
    setNotification(`"${book.title}" added to cart`);
    setTimeout(() => setNotification(null), 2500);
  };

  return (
    <div style={styles.body}>
      <div style={styles.marketContainer}>
        {notification && <div style={{ position: 'fixed', bottom: '24px', left: '50%', transform: 'translateX(-50%)', backgroundColor: '#1A1A1A', color: '#F6F4EC', padding: '12px 24px', fontSize: '12px', zIndex: 9999 }}>{notification}</div>}

        <header style={{ height: '60px', borderBottom: '1px solid #1A1A1A', display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', padding: '0 24px', backgroundColor: '#F6F4EC' }}>
          <nav style={{ display: 'flex', gap: '24px', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            <Link to="/catalog" style={{ textDecoration: 'none' }}>Catalog</Link>
            <Link to="/" style={{ textDecoration: 'none' }}>Home</Link>
          </nav>
          <div style={{ fontFamily: "'Cinzel', serif", fontSize: '24px' }}>Folio.</div>
          <nav style={{ display: 'flex', gap: '24px', justifyContent: 'flex-end', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', alignItems: 'center' }}>
            <UserSwitcher
              users={users}
              selectedUserId={selectedUserId}
              onChange={(userId) => {
                setSelectedUserId(userId);
                setStoredUserId(userId);
              }}
            />
            <span>Cart ({cartCount})</span>
          </nav>
        </header>

        <div style={{ minHeight: '60px', borderBottom: '1px solid #1A1A1A', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 24px', gap: '12px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {filters.map((f) => (
              <button
                key={f}
                onClick={() => {
                  setActiveFilter(f);
                  setPage(0);
                }}
                style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', padding: '8px 14px', border: activeFilter === f ? '1px solid #1A1A1A' : '1px solid transparent', borderRadius: activeFilter === f ? '99px' : '0', cursor: 'pointer', backgroundColor: 'transparent' }}
              >
                {f}
              </button>
            ))}
          </div>
          <input
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(0);
            }}
            placeholder="Search title"
            style={{ minWidth: '220px', border: '1px solid #1A1A1A', background: 'transparent', padding: '8px 10px', fontSize: '12px' }}
          />
        </div>

        <section style={{ padding: '20px 24px 8px 24px', borderBottom: '1px solid #1A1A1A' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h2 style={{ fontFamily: "'Cinzel', serif", fontSize: '22px', letterSpacing: '0.02em' }}>Top Recommendations</h2>
            <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>User: {selectedUserId || 'N/A'}</span>
          </div>
          {loadingRecs ? (
            <div style={{ fontSize: '12px', color: '#666' }}>Loading recommendations...</div>
          ) : topRecs.length === 0 ? (
            <div style={{ fontSize: '12px', color: '#666' }}>No recommendations for selected user.</div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(220px, 1fr))', gap: '32px 32px', rowGap: '64px' }}>
              {topRecs.map((book) => (
                <BookCard key={`top-${book.item_id}`} book={book} onAddToCart={onAddToCart} />
              ))}
            </div>
          )}
        </section>

        {error && (
          <div style={{ padding: '10px 24px', borderBottom: '1px solid #1A1A1A', fontSize: '12px', color: '#7f1d1d' }}>
            {error}
          </div>
        )}

        <section style={{ padding: '18px 24px 0 24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <h2 style={{ fontFamily: "'Cinzel', serif", fontSize: '22px', letterSpacing: '0.02em' }}>Main Catalog</h2>
            <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#666' }}>
              Browse all books
            </span>
          </div>
        </section>

        <main style={styles.grid}>
          {loading && books.length === 0 && (
            <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '80px 0', fontSize: '13px', color: '#666', textTransform: 'uppercase' }}>
              Loading catalog...
            </div>
          )}
          {!loading && books.map((book) => <BookCard key={book.id} book={book} onAddToCart={onAddToCart} />)}
          {!loading && books.length === 0 && <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '80px 0', fontSize: '13px', color: '#888', textTransform: 'uppercase' }}>No books found</div>}
        </main>

        <footer style={{ borderTop: '1px solid #1A1A1A', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 24px', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          <span>{total} books</span>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page <= 0}
              style={{ background: 'none', border: '1px solid #1A1A1A', padding: '6px 10px', cursor: page <= 0 ? 'not-allowed' : 'pointer', opacity: page <= 0 ? 0.4 : 1 }}
            >
              Prev
            </button>
            <span>Page {page + 1} / {totalPages}</span>
            <button
              onClick={() => setPage((p) => (p + 1 < totalPages ? p + 1 : p))}
              disabled={page + 1 >= totalPages}
              style={{ background: 'none', border: '1px solid #1A1A1A', padding: '6px 10px', cursor: page + 1 >= totalPages ? 'not-allowed' : 'pointer', opacity: page + 1 >= totalPages ? 0.4 : 1 }}
            >
              Next
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
