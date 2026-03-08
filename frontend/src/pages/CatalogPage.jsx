import React, { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

const styles = {
  body: { backgroundColor: '#EAE8E0', display: 'flex', justifyContent: 'center', minHeight: '100vh', padding: '20px' },
  marketContainer: { backgroundColor: '#F6F4EC', width: '100%', maxWidth: '1400px', border: '1px solid #1A1A1A', display: 'flex', flexDirection: 'column' },
  grid: { flex: 1, padding: '48px', display: 'grid', gridTemplateColumns: 'repeat(4, minmax(220px, 1fr))', gap: '32px 32px', rowGap: '64px', backgroundColor: '#F6F4EC' },
  bookWrapper: { perspective: '2000px', display: 'flex', justifyContent: 'center', alignItems: 'center', height: '380px', marginBottom: '18px' },
};

const books = [
  { id: 1, title: 'The Art of Silent Hours', titleLines: ['The Art of', 'Silent Hours'], author: 'Eleanor Voss', price: '$24.00', color: 'green', genre: 'Fiction' },
  { id: 2, title: 'Echoes of The Empire', titleLines: ['Echoes of', 'The Empire'], author: 'Marcus Harl', price: '$28.00', color: 'burgundy', genre: 'History' },
  { id: 3, title: 'Deep Water Navigation', titleLines: ['Deep Water', 'Navigation'], author: 'S. J. Thorne', price: '$32.00', color: 'navy', genre: 'Science' },
  { id: 4, title: 'Clay & Memory', titleLines: ['Clay &', 'Memory'], author: 'Ada L. Rose', price: '$22.00', color: 'terracotta', genre: 'Poetry' },
  { id: 5, title: 'Architects of Shadow', titleLines: ['Architects', 'of Shadow'], author: 'D. K. Vane', price: '$26.00', color: 'charcoal', genre: 'Fiction' },
  { id: 6, title: 'Wild Gardens', titleLines: ['Wild', 'Gardens'], author: 'Flora West', price: '$35.00', color: 'olive', genre: 'Essays' },
  { id: 7, title: 'Night Vision', titleLines: ['Night', 'Vision'], author: 'C. R. Stoker', price: '$21.00', color: 'purple', genre: 'Fiction' },
  { id: 8, title: 'The Cold Calculus', titleLines: ['The Cold', 'Calculus'], author: 'Ian M. Banks', price: '$29.00', color: 'blue', genre: 'Science' },
];

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

function BookCard({ book, onAddToCart }) {
  const [hovered, setHovered] = useState(false);
  const [cover, spine] = gradients[book.color];

  return (
    <article onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
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
              <div style={{ color: '#D4C89A', fontSize: '8px', letterSpacing: '0.15em', textTransform: 'uppercase', opacity: 0.8 }}>{book.author}</div>
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

      <div style={{ textAlign: 'center' }}>
        <h3 style={{ fontFamily: "'Cinzel', serif", marginBottom: '4px' }}>{book.title}</h3>
        <p style={{ fontSize: '10px', textTransform: 'uppercase', color: '#6b7280', marginBottom: '12px' }}>{book.author}</p>
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

  const filters = ['All', 'Fiction', 'Poetry', 'Essays', 'History', 'Science'];

  const filteredBooks = useMemo(
    () => (activeFilter === 'All' ? books : books.filter((b) => b.genre === activeFilter)),
    [activeFilter]
  );

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
          <nav style={{ display: 'flex', gap: '24px', justifyContent: 'flex-end', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            <span>Account</span>
            <span>Cart ({cartCount})</span>
          </nav>
        </header>

        <div style={{ minHeight: '60px', borderBottom: '1px solid #1A1A1A', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 24px', gap: '12px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {filters.map((f) => (
              <button key={f} onClick={() => setActiveFilter(f)} style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', padding: '8px 14px', border: activeFilter === f ? '1px solid #1A1A1A' : '1px solid transparent', borderRadius: activeFilter === f ? '99px' : '0', cursor: 'pointer', backgroundColor: 'transparent' }}>{f}</button>
            ))}
          </div>
        </div>

        <main style={styles.grid}>
          {filteredBooks.map((book) => <BookCard key={book.id} book={book} onAddToCart={onAddToCart} />)}
          {filteredBooks.length === 0 && <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '80px 0', fontSize: '13px', color: '#888', textTransform: 'uppercase' }}>No books found in this category</div>}
        </main>
      </div>
    </div>
  );
}
