import { Link } from "react-router-dom";

export default function Logo({ to = "/" }: { to?: string }) {
  return (
    <Link className="logo" to={to} aria-label="CiteWise home">
      <svg viewBox="0 0 36 36" aria-hidden="true">
        <path d="M4 8c5 0 9 1.5 14 5v18c-5-3.5-9-5-14-5V8Z" />
        <path d="M32 8c-5 0-9 1.5-14 5v18c5-3.5 9-5 14-5V8Z" />
      </svg>
      <span>CiteWise</span>
    </Link>
  );
}
