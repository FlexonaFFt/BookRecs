import React, { useState } from 'react';
import { Link } from 'react-router-dom';

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

function SpecItem({ label, value, last }) {
  return (
    <div style={last ? styles.specItemLast : styles.specItem}>
      <span style={styles.specLabel}>{label}</span>
      <span style={styles.specValue}>{value}</span>
    </div>
  );
}

function Book3D({ hovered }) {
  return (
    <div
      style={{
        ...styles.scene,
        transform: hovered ? 'scale(1) rotateY(-15deg) rotateX(6deg)' : 'scale(1) rotateY(-25deg) rotateX(10deg)',
      }}
    >
      <div style={styles.book}>
        <div style={styles.bookCover}>
          <div style={styles.bookCoverOrnament}></div>
          <div style={styles.bookCoverTitle}>The Art of<br />Silent Hours</div>
          <div style={styles.bookCoverRule}></div>
          <div style={styles.bookCoverAuthor}>Eleanor Voss</div>
        </div>
        <div style={styles.bookSpine}><div style={styles.bookSpineText}>The Art of Silent Hours</div></div>
        <div style={styles.bookBack}></div>
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

  return (
    <div style={styles.body}>
      <div style={styles.cardFrame}>
        <header style={styles.cardHeader}>
          <div style={styles.nav}><Link to="/catalog">Catalog</Link><span>About</span></div>
          <div style={styles.brand}>Folio.</div>
          <div style={{ ...styles.nav, justifyContent: 'flex-end' }}><span>Account</span><span>Cart (0)</span></div>
        </header>

        <div style={styles.cardBody}>
          <div style={styles.contentCol}>
            <div style={styles.circleBadge}>New<br />Arrivals</div>
            <h1 style={styles.h1}>Stories worth<br />holding in<br />your hands</h1>
            <p style={styles.description}>Curated editions, rare finds, and modern classics. Every book in our collection is chosen for its craft.</p>
            <Link
              to="/catalog"
              style={{ ...styles.ctaBtn, opacity: ctaHovered ? 0.6 : 1 }}
              onMouseEnter={() => setCtaHovered(true)}
              onMouseLeave={() => setCtaHovered(false)}
            >
              Browse the collection
            </Link>
          </div>

          <div style={styles.visualCol} onMouseEnter={() => setIsHovered(true)} onMouseLeave={() => setIsHovered(false)}>
            <Book3D hovered={isHovered} />
          </div>
        </div>

        <footer style={styles.cardFooter}>
          <SpecItem label="Genre" value="Literary Fiction" />
          <SpecItem label="Format" value="Hardcover" />
          <SpecItem label="Pages" value="348" />
          <SpecItem label="Language" value="English" last />
        </footer>
      </div>
    </div>
  );
}
