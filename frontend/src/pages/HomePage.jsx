import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import UserSwitcher from '../components/UserSwitcher';
import { extractPartLabel, formatDisplayTitle, splitTitleForCover } from '../utils/bookFormat';
import { addCartItem, getCartCount, onCartUpdate } from '../utils/cartStore';
import {
  fetchDemoBook,
  fetchDemoUsers,
  postInteraction,
  fetchRecommendations,
  getStoredUserId,
  setStoredUserId,
} from '../api/demoApi';

const fallbackBook = {
  item_id: 1,
  title: 'The Art of Silent Hours',
  authors: ['Eleanor Voss'],
  tags: ['fiction'],
  series: [],
};

const styles = {
  body: {
    backgroundColor: '#EAE8E0',
    color: '#1A1A1A',
    fontFamily: "'Inter', sans-serif",
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px',
    WebkitFontSmoothing: 'antialiased',
  },
  cardFrame: {
    backgroundColor: '#F6F4EC',
    width: '100%',
    maxWidth: '1400px',
    minHeight: 'calc(100vh - 40px)',
    border: '1px solid #1A1A1A',
    display: 'grid',
    gridTemplateRows: 'auto 1fr auto',
    boxShadow: '0 20px 40px rgba(0,0,0,0.05)',
  },
  cardHeader: {
    minHeight: '60px',
    borderBottom: '1px solid #1A1A1A',
    display: 'grid',
    gridTemplateColumns: '1fr auto 1fr',
    alignItems: 'center',
    padding: '0 24px',
  },
  brand: { fontFamily: "'Cinzel', serif", fontSize: '24px', textAlign: 'center', gridColumn: 2 },
  nav: {
    fontSize: '10px',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    display: 'flex',
    gap: '24px',
    alignItems: 'center',
  },
  cardBody: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    overflow: 'hidden',
  },
  contentCol: {
    borderRight: '1px solid #1A1A1A',
    padding: '60px 40px',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    position: 'relative',
  },
  visualCol: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    perspective: '2000px',
    padding: '24px',
  },
  h1: {
    fontWeight: 500,
    fontSize: 'clamp(28px, 3vw, 56px)',
    textTransform: 'uppercase',
    lineHeight: 1.2,
    marginBottom: '24px',
    maxWidth: '420px',
  },
  description: {
    fontSize: '20px',
    lineHeight: 1.6,
    maxWidth: '500px',
    marginBottom: '40px',
    color: '#333',
  },
  circleBadge: {
    width: '100px',
    height: '100px',
    border: '1px solid #1A1A1A',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
    fontSize: '10px',
    textTransform: 'uppercase',
    position: 'absolute',
    top: '40px',
    right: '40px',
    transform: 'rotate(-15deg)',
    letterSpacing: '0.04em',
  },
  ctaBtn: {
    fontSize: '11px',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    textDecoration: 'none',
    borderBottom: '1px solid #1A1A1A',
    paddingBottom: '2px',
    alignSelf: 'flex-start',
  },
  cardFooter: {
    minHeight: '100px',
    borderTop: '1px solid #1A1A1A',
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
  },
  specItem: {
    borderRight: '1px solid #1A1A1A',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    textAlign: 'center',
    padding: '16px',
  },
  specItemLast: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    textAlign: 'center',
    padding: '16px',
  },
  specLabel: { fontSize: '10px', textTransform: 'uppercase', marginBottom: '8px', color: '#666', letterSpacing: '0.08em' },
  specValue: { fontSize: '13px', fontWeight: 500 },
  scene: { transformStyle: 'preserve-3d', transition: 'transform 0.5s ease-out' },
  book: { position: 'relative', width: '220px', height: '300px', transformStyle: 'preserve-3d' },
  bookCover: {
    position: 'absolute',
    width: '220px',
    height: '300px',
    background: 'linear-gradient(160deg, #3B5249 0%, #2A3D33 100%)',
    transform: 'translateZ(30px)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '28px 22px',
    boxShadow: 'inset -4px 0 12px rgba(0,0,0,0.25)',
  },
  bookCoverOrnament: {
    position: 'absolute',
    top: '18px',
    right: '18px',
    width: '32px',
    height: '32px',
    border: '1px solid #D4C89A',
    borderRadius: '50%',
    opacity: 0.4,
  },
  bookCoverTitle: {
    fontFamily: "'Cinzel', serif",
    color: '#D4C89A',
    fontSize: '18px',
    textAlign: 'center',
    letterSpacing: '0.08em',
    lineHeight: 1.4,
    marginBottom: '16px',
    textTransform: 'uppercase',
  },
  bookCoverRule: {
    width: '60px',
    height: '1px',
    background: '#D4C89A',
    marginBottom: '16px',
    opacity: 0.6,
  },
  bookCoverAuthor: {
    color: '#D4C89A',
    fontSize: '10px',
    letterSpacing: '0.18em',
    textTransform: 'uppercase',
    opacity: 0.7,
  },
  bookSpine: {
    position: 'absolute',
    width: '60px',
    height: '300px',
    background: 'linear-gradient(90deg, #1E2B24 0%, #2A3D33 100%)',
    transform: 'rotateY(-90deg) translateZ(30px)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
  },
  bookSpineText: {
    fontFamily: "'Cinzel', serif",
    color: '#D4C89A',
    fontSize: '11px',
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    writingMode: 'vertical-rl',
    transform: 'rotate(180deg)',
    opacity: 0.85,
  },
  bookBack: {
    position: 'absolute',
    width: '220px',
    height: '300px',
    background: '#1E2B24',
    transform: 'translateZ(-30px) rotateY(180deg)',
  },
  bookPages: {
    position: 'absolute',
    width: '60px',
    height: '300px',
    background: 'repeating-linear-gradient(to bottom, #F6F4EC 0px, #F6F4EC 1px, #E8E5DB 2px, #E8E5DB 3px)',
    transform: 'rotateY(90deg) translateZ(190px)',
    boxShadow: 'inset -4px 0 8px rgba(0,0,0,0.08)',
  },
  bookTop: {
    position: 'absolute',
    width: '220px',
    height: '60px',
    background: 'linear-gradient(180deg, #E8E5DB 0%, #D8D5CB 100%)',
    transform: 'rotateX(90deg) translateZ(300px)',
  },
  bookBottom: {
    position: 'absolute',
    width: '220px',
    height: '60px',
    background: '#C8C5BB',
    transform: 'rotateX(-90deg) translateZ(0px)',
    boxShadow: '0 40px 80px rgba(0,0,0,0.22)',
  },
};

const bookPalettes = {
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
const paletteKeys = Object.keys(bookPalettes);

function firstOf(list, fallback = '') {
  return Array.isArray(list) && list.length > 0 ? String(list[0]) : fallback;
}

function splitTitle(title) {
  return splitTitleForCover(title);
}

function SpecItem({ label, value, last }) {
  return (
    <div style={last ? styles.specItemLast : styles.specItem}>
      <span style={styles.specLabel}>{label}</span>
      <span style={styles.specValue}>{value}</span>
    </div>
  );
}

function Book3D({ hovered, title, partLabel, palette }) {
  const parts = splitTitle(title);
  const displayTitle = formatDisplayTitle(title);
  const colors = bookPalettes[palette] || bookPalettes.green;
  const coverStyle = { ...styles.bookCover, background: colors.cover };
  const spineStyle = { ...styles.bookSpine, background: colors.spine };
  const backStyle = { ...styles.bookBack, background: colors.back };
  return (
    <div
      style={{
        ...styles.scene,
        transform: hovered ? 'scale(1) rotateY(-15deg) rotateX(6deg)' : 'scale(1) rotateY(-25deg) rotateX(10deg)',
      }}
    >
      <div style={styles.book}>
        <div style={coverStyle}>
          <div style={styles.bookCoverOrnament}></div>
          <div style={styles.bookCoverTitle}>{parts[0]}<br />{parts[1]}</div>
          <div style={styles.bookCoverRule}></div>
          <div style={styles.bookCoverAuthor}>{partLabel}</div>
        </div>
        <div style={spineStyle}><div style={styles.bookSpineText}>{displayTitle}</div></div>
        <div style={backStyle}></div>
        <div style={styles.bookPages}></div>
        <div style={styles.bookTop}></div>
        <div style={styles.bookBottom}></div>
      </div>
    </div>
  );
}

export default function HomePage() {
  const [isHovered, setIsHovered] = useState(false);
  const [ctaHovered, setCtaHovered] = useState(false);
  const [cartCount, setCartCount] = useState(0);
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState('');
  const [recommendedBook, setRecommendedBook] = useState(fallbackBook);
  const [loadingReco, setLoadingReco] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function loadUsers() {
      try {
        const list = await fetchDemoUsers(200);
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
    async function loadTop1Recommendation() {
      if (!selectedUserId) {
        return;
      }
      setLoadingReco(true);
      try {
        const reco = await fetchRecommendations(selectedUserId, 1);
        const top = Array.isArray(reco.items) && reco.items.length > 0 ? reco.items[0] : null;
        if (!top) {
          throw new Error('empty top-1');
        }
        const book = await fetchDemoBook(top.item_id);
        if (!cancelled && book) {
          setRecommendedBook(book);
        }
      } catch {
        if (!cancelled) {
          setRecommendedBook(fallbackBook);
        }
      } finally {
        if (!cancelled) {
          setLoadingReco(false);
        }
      }
    }
    loadTop1Recommendation();
    return () => {
      cancelled = true;
    };
  }, [selectedUserId]);

  const partLabel = extractPartLabel(recommendedBook.title);
  const series = firstOf(recommendedBook.series, 'Standalone');
  const price = `$${20 + (Number(recommendedBook.item_id || 0) % 16)}.00`;
  const displayTitle = formatDisplayTitle(recommendedBook.title);
  const palette = paletteKeys[Math.abs(Number(recommendedBook.item_id || 0)) % paletteKeys.length];

  const pageCount = useMemo(() => {
    const text = String(recommendedBook.description || '');
    return Math.max(120, Math.min(650, Math.floor(text.length / 4) || 220));
  }, [recommendedBook.description]);

  return (
    <div style={styles.body}>
      <div style={styles.cardFrame}>
        <header style={styles.cardHeader}>
          <div style={styles.nav}><Link to="/catalog">Catalog</Link><span>About</span></div>
          <Link to="/" style={{ ...styles.brand, textDecoration: 'none', color: 'inherit' }}>BookRec.</Link>
          <div style={{ ...styles.nav, justifyContent: 'flex-end' }}>
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

        <div style={styles.cardBody}>
          <div style={styles.contentCol}>
            <div style={styles.circleBadge}>Top 1<br />for today</div>
            <h1 style={styles.h1}>This is your<br />recommendation<br />for today</h1>
            <p style={styles.description}>
              {loadingReco ? 'Preparing your personalized pick...' : `Today we recommend "${displayTitle}".`}
            </p>
            <div style={{ display: 'flex', gap: '18px', alignItems: 'center', marginBottom: '24px' }}>
              <span style={{ fontSize: '18px', fontWeight: 500 }}>{price}</span>
              <button
                onClick={async () => {
                  addCartItem(selectedUserId, {
                    item_id: recommendedBook.item_id,
                    title: displayTitle,
                    partLabel,
                    price,
                  });
                  try {
                    await postInteraction({
                      userId: selectedUserId,
                      itemId: recommendedBook.item_id,
                      eventType: 'add_to_cart',
                    });
                  } catch {
                    // keep demo smooth even if logging fails
                  }
                }}
                style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.08em', background: 'none', border: 'none', borderBottom: '1px solid #1A1A1A', cursor: 'pointer' }}
              >
                Add to cart
              </button>
            </div>
            <Link
              to="/catalog"
              style={{ ...styles.ctaBtn, opacity: ctaHovered ? 0.6 : 1 }}
              onMouseEnter={() => setCtaHovered(true)}
              onMouseLeave={() => setCtaHovered(false)}
            >
              See top 10 recommendations
            </Link>
          </div>

          <div style={styles.visualCol} onMouseEnter={() => setIsHovered(true)} onMouseLeave={() => setIsHovered(false)}>
            <Link to={`/book/${recommendedBook.item_id}`} style={{ textDecoration: 'none' }}>
              <Book3D hovered={isHovered} title={recommendedBook.title} partLabel={partLabel} palette={palette} />
            </Link>
          </div>
        </div>

        <footer style={styles.cardFooter}>
          <SpecItem label="Book ID" value={String(recommendedBook.item_id || 'N/A')} />
          <SpecItem label="Series" value={series} />
          <SpecItem label="Pages" value={String(pageCount)} />
          <SpecItem label="Part" value={partLabel} last />
        </footer>
      </div>
    </div>
  );
}
