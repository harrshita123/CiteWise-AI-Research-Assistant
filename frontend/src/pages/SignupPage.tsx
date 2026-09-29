import { FormEvent, useCallback, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AuthLayout from "../components/AuthLayout";
import GoogleSignInButton from "../components/GoogleSignInButton";
import PasswordInput from "../components/PasswordInput";
import { signInWithGoogle, signUp, SignedInUser } from "../services/authApi";

export default function SignupPage() {
  const navigate = useNavigate();
  const [user, setUser] = useState<SignedInUser | null>(null);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleCredential = useCallback(async (credential: string) => {
    const { user } = await signInWithGoogle(credential);
    setUser(user);
    navigate("/home", { replace: true });
  }, [navigate]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    const form = new FormData(event.currentTarget);
    const password = String(form.get("password") ?? "");
    const confirmPassword = String(form.get("confirmPassword") ?? "");

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setIsSubmitting(true);

    try {
      const result = await signUp(
        String(form.get("name") ?? ""),
        String(form.get("email") ?? ""),
        password,
      );
      setUser(result.user);
      navigate("/home", { replace: true });
    } catch (error) {
      setError(error instanceof Error ? error.message : "Account creation failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AuthLayout
      title="Create your account"
      description="Your research workspace starts here."
      footer={<>Already have an account? <Link to="/login">Log in</Link></>}
    >
      {user ? (
        <div className="auth-form">
          <p className="form-message" role="status">Your CiteWise account is ready for {user.email}.</p>
          <Link className="primary-button" to="/home">Open home</Link>
        </div>
      ) : (
        <div className="auth-form">
          <form className="credential-form" onSubmit={handleSubmit}>
            <div className="field">
              <label htmlFor="signup-name">Full name</label>
              <input id="signup-name" name="name" autoComplete="name" minLength={2} maxLength={80} required />
            </div>
            <div className="field">
              <label htmlFor="signup-email">Email address</label>
              <input id="signup-email" name="email" type="email" autoComplete="email" required />
            </div>
            <PasswordInput id="signup-password" name="password" label="Password" autoComplete="new-password" />
            <PasswordInput id="confirm-password" name="confirmPassword" label="Confirm password" autoComplete="new-password" />
            <button className="primary-button" type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Creating account..." : "Create account"}
            </button>
          </form>

          <div className="auth-divider"><span>or</span></div>
          <GoogleSignInButton text="signup_with" onCredential={handleCredential} onError={setError} />
          {error && <p className="oauth-error" role="alert">{error}</p>}
        </div>
      )}
    </AuthLayout>
  );
}
