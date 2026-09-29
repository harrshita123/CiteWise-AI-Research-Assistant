import { useEffect, useRef, useState } from "react";
import { getGoogleClientId, loadGoogleIdentity } from "../services/googleIdentity";

type Props = {
  text: "signin_with" | "signup_with";
  onCredential: (credential: string) => Promise<void>;
  onError: (message: string) => void;
};

export default function GoogleSignInButton({ text, onCredential, onError }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    const clientId = getGoogleClientId();

    if (!clientId) {
      onError("Google sign-in is not configured yet. Add VITE_GOOGLE_CLIENT_ID to frontend/.env.");
      return;
    }

    let isActive = true;

    async function renderButton() {
      try {
        const google = await loadGoogleIdentity();

        if (!isActive || !container.current) {
          return;
        }

        google.initialize({
          client_id: clientId,
          callback: async ({ credential }) => {
            onError("");
            setIsSubmitting(true);

            try {
              await onCredential(credential);
            } catch (error) {
              const message = error instanceof Error ? error.message : "Google sign-in could not be completed.";
              onError(message);
            } finally {
              setIsSubmitting(false);
            }
          },
        });

        container.current.replaceChildren();
        google.renderButton(container.current, {
          theme: "outline",
          size: "medium",
          shape: "rectangular",
          text,
          width: container.current.clientWidth || 360,
        });
      } catch (error) {
        if (isActive) {
          const message = error instanceof Error ? error.message : "Google sign-in could not be loaded.";
          onError(message);
        }
      }
    }

    void renderButton();

    return () => {
      isActive = false;
    };
  }, [onCredential, onError, text]);

  return (
    <div className="google-signin">
      <div className={isSubmitting ? "google-signin-button is-loading" : "google-signin-button"} ref={container} />
      {isSubmitting && <p className="oauth-status" role="status">Signing you in securely...</p>}
    </div>
  );
}
