type IconName = "home" | "library" | "search" | "compare" | "shield" | "upload" | "arrow" | "chevron" | "file" | "clock" | "check" | "warning" | "logout" | "user";

export default function WorkspaceIcon({ name, size = 22 }: { name: IconName; size?: number }) {
  const shared = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true as const };
  const paths: Record<IconName, React.ReactNode> = {
    home: <><path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V10Z" /><path d="M9 21v-7h6v7" /></>,
    library: <><rect x="4" y="2.5" width="16" height="19" rx="2" /><path d="M8 7h8M8 11h8M8 15h5" /></>,
    search: <><circle cx="10.8" cy="10.8" r="7" /><path d="m16 16 5 5" /></>,
    compare: <><path d="M12 3v18M5 6h14M5 6 2 12h6L5 6Zm14 0-3 6h6l-3-6ZM3 16c.5 2 3.5 2 4 0M17 16c.5 2 3.5 2 4 0" /></>,
    shield: <><path d="M12 2 4 5v6c0 5 3 8 8 11 5-3 8-6 8-11V5l-8-3Z" /><path d="m8.5 12 2.5 2.5 4.5-5" /></>,
    upload: <><path d="M7 17.5a4.5 4.5 0 0 1-1-8.9A6.5 6.5 0 0 1 18.4 11 3.5 3.5 0 0 1 18 18h-2" /><path d="M12 22V12m-4 4 4-4 4 4" /></>,
    arrow: <><path d="M4 12h16m-6-6 6 6-6 6" /></>,
    chevron: <><path d="m9 5 7 7-7 7" /></>,
    file: <><path d="M6 2h8l4 4v15a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1Z" /><path d="M14 2v5h4M8 12h7m-7 4h7" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    check: <><circle cx="12" cy="12" r="9" /><path d="m8 12 3 3 5-6" /></>,
    warning: <><circle cx="12" cy="12" r="9" /><path d="M12 7v6m0 4h.01" /></>,
    logout: <><path d="M9 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h4M15 16l4-4-4-4m4 4H8" /></>,
    user: <><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></>,
  };
  return <svg {...shared}>{paths[name]}</svg>;
}
