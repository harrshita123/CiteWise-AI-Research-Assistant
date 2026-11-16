export type Theme = "light" | "dark";
export function readTheme(storage?: Pick<Storage, "getItem">): Theme;
export function applyTheme(theme: Theme, root?: HTMLElement, storage?: Pick<Storage, "setItem">): void;
