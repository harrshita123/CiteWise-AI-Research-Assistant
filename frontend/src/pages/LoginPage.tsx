import { FormEvent, useCallback, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import AuthLayout from "../components/AuthLayout";
import GoogleSignInButton from "../components/GoogleSignInButton";
import PasswordInput from "../components/PasswordInput";
import { logIn, signInWithGoogle, SignedInUser } from "../services/authApi";

export default function LoginPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const destination = params.get("next") === "/research" ? "/research" : "/home";
  const [user, setUser] = useState<SignedInUser | null>(null);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleCredential = useCallback(async (credential: string) => {
    const { user } = await signInWithGoogle(credential);
    setUser(user);
    navigate(destination, { replace: true });
  }, [navigate, destination]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsSubmitting(true);

    const form = new FormData(event.currentTarget);

    try {
      const result = await logIn(
        String(form.get("email") ?? ""),
        String(form.get("password") ?? ""),
      );
      setUser(result.user);
      navigate(destination, { replace: true });
    } catch (error) {
      setError(error instanceof Error ? error.message : "Login failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AuthLayout
      title="Welcome back"
      description="Continue your research with evidence you can verify."
      footer={<>New to CiteWise? <Link to="/signup">Create an account</Link></>}
    >
      {user ? (
        <div className="auth-form">
          <p className="form-message" role="status">
            {user.role === "admin" ? "Admin login successful" : "Login successful"}. Signed in as {user.email}.
          </p>
          <Link className="primary-button" to="/home">
            Open home
          </Link>
        </div>
      ) : (
        <div className="auth-form">
          <form className="credential-form" onSubmit={handleSubmit}>
            <div className="field">
              <label htmlFor="login-email">Email address</label>
              <input id="login-email" name="email" type="email" autoComplete="email" required />
            </div>
            <PasswordInput id="login-password" name="password" label="Password" autoComplete="current-password" />
            <button className="primary-button" type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Logging in..." : "Log in"}
            </button>
          </form>

          <div className="auth-divider"><span>or</span></div>
          <GoogleSignInButton text="signin_with" onCredential={handleCredential} onError={setError} />
          {error && <p className="oauth-error" role="alert">{error}</p>}
        </div>
      )}
    </AuthLayout>
  );
}
