export const DEFAULT_SIDEBAR_WIDTH = 300;
export const MIN_SIDEBAR_WIDTH = 220;
export const MAX_SIDEBAR_WIDTH = 420;

export function clampSidebarWidth(value) {
  const width = Number(value);
  return Number.isFinite(width) ? Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, width)) : DEFAULT_SIDEBAR_WIDTH;
}

export function resizedSidebarWidth(startWidth, startX, currentX) {
  return clampSidebarWidth(startWidth + currentX - startX);
}
