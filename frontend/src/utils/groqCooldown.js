export function deadlineFromRetry(seconds, now = Date.now()) {
  return Number.isFinite(seconds) && seconds > 0 ? now + Math.ceil(seconds) * 1000 : 0;
}

export function remainingSeconds(deadline, now = Date.now()) {
  return Math.max(0, Math.ceil((deadline - now) / 1000));
}
