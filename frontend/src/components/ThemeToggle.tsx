import { useEffect, useState } from "react";
import WorkspaceIcon from "./WorkspaceIcon";
import { applyTheme, readTheme, type Theme } from "../utils/theme";

export default function ThemeToggle({ className = "" }: { className?: string }) {
  const [theme, setTheme] = useState<Theme>(() => readTheme());
  useEffect(() => { applyTheme(theme); }, [theme]);
  const next = theme === "dark" ? "light" : "dark";

  return <button className={className} type="button" aria-label={`Switch to ${next} theme`}
    title={`Switch to ${next} theme`} onClick={() => setTheme(next)}>
    <WorkspaceIcon name={theme === "dark" ? "sun" : "moon"} size={18} />
  </button>;
}
