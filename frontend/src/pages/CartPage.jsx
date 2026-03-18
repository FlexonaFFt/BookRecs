import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import UserSwitcher from '../components/UserSwitcher';
import { fetchDemoUsers, getStoredUserId, postInteraction, setStoredUserId } from '../api/demoApi';
import { clearCart, getCartItems, onCartUpdate, removeCartItem } from '../utils/cartStore';

export default function CartPage() {
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState('');
  const [items, setItems] = useState([]);
  const [status, setStatus] = useState('');

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
    function refresh() {
      setItems(getCartItems(selectedUserId));
    }
    refresh();
    const off = onCartUpdate(() => refresh());
    return off;
  }, [selectedUserId]);

  const total = useMemo(() => {
    return items.reduce((acc, item) => {
      const n = Number(String(item.price || '0').replace('$', ''));
      return acc + (Number.isFinite(n) ? n : 0);
    }, 0);
  }, [items]);

  async function onPurchase() {
    if (!selectedUserId || items.length === 0) return;
    try {
      await Promise.all(
        items.map((item) =>
          postInteraction({
            userId: selectedUserId,
            itemId: item.item_id,
            eventType: 'purchase',
          })
        )
      );
      clearCart(selectedUserId);
      setStatus('Purchase completed');
    } catch {
      clearCart(selectedUserId);
      setStatus('Purchase completed (offline)');
    }
    setTimeout(() => setStatus(''), 2200);
  }

  return (
    <div style={{ backgroundColor: '#EAE8E0', minHeight: '100vh', padding: '20px' }}>
      <div style={{ backgroundColor: '#F6F4EC', border: '1px solid #1A1A1A', maxWidth: '1200px', margin: '0 auto' }}>
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
            <span>Cart ({items.length})</span>
          </div>
        </header>

        <section style={{ padding: '24px', borderBottom: '1px solid #1A1A1A' }}>
          <h1 style={{ fontFamily: "'Cinzel', serif", fontSize: '36px', marginBottom: '10px' }}>Cart</h1>
          <p style={{ fontSize: '12px', color: '#666', textTransform: 'uppercase' }}>Current account: {selectedUserId || 'N/A'}</p>
        </section>

        <section style={{ padding: '24px', borderBottom: '1px solid #1A1A1A' }}>
          {items.length === 0 ? (
            <div style={{ fontSize: '14px', color: '#666' }}>Cart is empty.</div>
          ) : (
            <div style={{ display: 'grid', gap: '12px' }}>
              {items.map((item, idx) => (
                <article key={`${item.item_id}-${idx}`} style={{ border: '1px solid #1A1A1A', padding: '12px', display: 'grid', gridTemplateColumns: '1fr auto auto', gap: '12px', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontFamily: "'Cinzel', serif", fontSize: '22px' }}>{item.title}</div>
                    <div style={{ fontSize: '10px', textTransform: 'uppercase', color: '#6b7280' }}>{item.partLabel}</div>
                  </div>
                  <div style={{ fontSize: '22px', fontWeight: 500 }}>{item.price}</div>
                  <button
                    onClick={() => removeCartItem(selectedUserId, idx)}
                    style={{ background: 'transparent', border: '1px solid #1A1A1A', padding: '8px 10px', textTransform: 'uppercase', fontSize: '10px', cursor: 'pointer' }}
                  >
                    Remove
                  </button>
                </article>
              ))}
            </div>
          )}
        </section>

        <section style={{ padding: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Total: <strong>${total.toFixed(2)}</strong></div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              onClick={() => clearCart(selectedUserId)}
              style={{ background: 'transparent', border: '1px solid #1A1A1A', padding: '10px 14px', textTransform: 'uppercase', fontSize: '11px', cursor: 'pointer' }}
            >
              Clear cart
            </button>
            <button
              onClick={onPurchase}
              disabled={items.length === 0}
              style={{ background: '#1A1A1A', color: '#F6F4EC', border: 'none', padding: '10px 14px', textTransform: 'uppercase', fontSize: '11px', cursor: items.length === 0 ? 'not-allowed' : 'pointer', opacity: items.length === 0 ? 0.5 : 1 }}
            >
              Complete purchase
            </button>
          </div>
        </section>

        {status && <div style={{ padding: '0 24px 24px', fontSize: '12px', textTransform: 'uppercase', color: '#334155' }}>{status}</div>}
      </div>
    </div>
  );
}
