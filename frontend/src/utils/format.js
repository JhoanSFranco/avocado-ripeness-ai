/** Remaining time as "N hours" (under a day) or "N.N days". */
export function formatDuration(days) {
  if (days === null || days === undefined || Number.isNaN(Number(days))) return '-';
  const value = Number(days);
  if (value < 1) {
    const hours = Math.max(1, Math.round(value * 24));
    return `${hours} hour${hours === 1 ? '' : 's'}`;
  }
  return `${value.toFixed(1)} days`;
}

export const formatPercent = (probability) => `${Math.round((probability || 0) * 100)}%`;

/** Random id accepted by the API (`^[A-Za-z0-9_-]{1,64}$`).
 *  crypto.randomUUID is only available on HTTPS/localhost, hence the fallback. */
export function newSessionId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}
