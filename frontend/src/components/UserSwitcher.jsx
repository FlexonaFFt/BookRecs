import { useEffect, useMemo, useRef, useState } from 'react';

export default function UserSwitcher({ users, selectedUserId, onChange }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    function handleOutsideClick(event) {
      if (!rootRef.current) {
        return;
      }
      if (!rootRef.current.contains(event.target)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  const items = useMemo(
    () =>
      users.map((u, idx) => ({
        userId: u.user_id,
        label: `account${idx + 1}`,
      })),
    [users]
  );
  const visibleItems = items.slice(0, 15);

  return (
    <div ref={rootRef} style={{ position: 'relative' }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          background: 'transparent',
          fontSize: '10px',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          border: 'none',
          borderBottom: '1px solid #1A1A1A',
          padding: '2px 0',
          cursor: 'pointer',
        }}
      >
        Account
      </button>

      {open && (
        <div
          style={{
            position: 'absolute',
            right: 0,
            top: 'calc(100% + 8px)',
            minWidth: '170px',
            border: '1px solid #1A1A1A',
            backgroundColor: '#F6F4EC',
            zIndex: 3000,
            maxHeight: '360px',
            overflowY: 'auto',
          }}
        >
          {items.length === 0 && (
            <div style={{ padding: '10px 12px', fontSize: '10px', textTransform: 'uppercase' }}>
              No accounts
            </div>
          )}
          {visibleItems.map((item) => {
            const active = item.userId === selectedUserId;
            return (
              <button
                key={item.userId}
                type="button"
                onClick={() => {
                  onChange(item.userId);
                  setOpen(false);
                }}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  width: '100%',
                  background: active ? 'rgba(26,26,26,0.06)' : 'transparent',
                  border: 'none',
                  borderBottom: '1px solid rgba(26,26,26,0.12)',
                  padding: '10px 12px',
                  textAlign: 'left',
                  fontSize: '10px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  cursor: 'pointer',
                }}
              >
                <span>{item.label}</span>
                {active && <span style={{ fontSize: '9px', opacity: 0.7 }}>selected</span>}
              </button>
            );
          })}
          {items.length > 15 && (
            <div style={{ padding: '8px 12px', fontSize: '9px', textTransform: 'uppercase', color: '#666' }}>
              Showing first 15 accounts
            </div>
          )}
        </div>
      )}
    </div>
  );
}
