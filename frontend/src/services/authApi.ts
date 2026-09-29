export type SignedInUser = {
  id: string;
  email: string;
  name: string | null;
  role: "user" | "admin";
};

type AuthenticationResponse = {
  user: SignedInUser;
};

export class AuthApiError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message);
  }
}

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${apiBaseUrl}${path}`, {
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      ...options,
    });
  } catch {
    throw new Error("Authentication service is unavailable. Start the backend and try again.");
  }

  const payload = await response.json().catch(() => null) as { detail?: string } | AuthenticationResponse | null;

  if (!response.ok) {
    const detail = payload && "detail" in payload ? payload.detail : null;
    const fallback = response.status >= 500
      ? "Authentication service is unavailable. Start the backend and try again."
      : "Authentication could not be completed.";
    throw new AuthApiError(detail || fallback, response.status);
  }

  return payload as T;
}

function postAuthentication(path: string, body: object): Promise<AuthenticationResponse> {
  return request(`/api/v1/auth/${path}`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function signUp(name: string, email: string, password: string) {
  return postAuthentication("signup", { name, email, password });
}

export function logIn(email: string, password: string) {
  return postAuthentication("login", { email, password });
}

export function signInWithGoogle(credential: string) {
  return postAuthentication("google", { credential });
}

export function getCurrentUser() {
  return request<SignedInUser>("/api/v1/auth/me");
}

export function getAdminUsers() {
  return request<SignedInUser[]>("/api/v1/admin/users");
}

export function logOut() {
  return request<null>("/api/v1/auth/logout", { method: "POST" });
}
