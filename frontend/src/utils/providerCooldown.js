export function deadlineFromRetry(seconds, now = Date.now()) {
  return Number.isFinite(seconds) && seconds > 0 ? now + Math.ceil(seconds) * 1000 : 0;
}

export function remainingSeconds(deadline, now = Date.now()) {
  return Math.max(0, Math.ceil((deadline - now) / 1000));
}

export function formatRetryDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "";
  if (seconds < 60) return `${Math.ceil(seconds)} s`;
  const minutes = Math.ceil(seconds / 60);
  const days = Math.floor(minutes / 1440);
  const hours = Math.floor((minutes % 1440) / 60);
  const remainder = minutes % 60;
  return [days && `${days} d`, hours && `${hours} h`, remainder && `${remainder} min`]
    .filter(Boolean).join(" ");
}
