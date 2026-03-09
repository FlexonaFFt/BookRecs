import React from 'react';

export default function UserSwitcher({ users, selectedUserId, onChange }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <span>Account</span>
      <select
        value={selectedUserId || ''}
        onChange={(e) => onChange(e.target.value)}
        style={{
          border: '1px solid #1A1A1A',
          background: 'transparent',
          fontSize: '10px',
          padding: '4px 6px',
          textTransform: 'none',
          maxWidth: '150px',
        }}
      >
        {users.length === 0 && <option value="">No users</option>}
        {users.map((u) => (
          <option key={u.user_id} value={u.user_id}>
            {u.user_id}
          </option>
        ))}
      </select>
    </div>
  );
}
