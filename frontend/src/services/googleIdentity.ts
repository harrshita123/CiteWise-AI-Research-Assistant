const googleScriptId = "google-identity-services";
const googleScriptUrl = "https://accounts.google.com/gsi/client";
let googleIdentityPromise: Promise<GoogleIdentityApi> | null = null;

export function getGoogleClientId(): string {
  return (import.meta.env.VITE_GOOGLE_CLIENT_ID ?? "").trim();
}

export async function loadGoogleIdentity(): Promise<GoogleIdentityApi> {
  if (window.google?.accounts.id) {
    return window.google.accounts.id;
  }

  if (!googleIdentityPromise) {
    googleIdentityPromise = new Promise<GoogleIdentityApi>((resolve, reject) => {
      const existingScript = document.getElementById(googleScriptId) as HTMLScriptElement | null;
      const script = existingScript ?? document.createElement("script");

      function finishLoading() {
        if (window.google?.accounts.id) {
          resolve(window.google.accounts.id);
          return;
        }

        reject(new Error("Google sign-in is unavailable."));
      }

      script.addEventListener("load", finishLoading, { once: true });
      script.addEventListener("error", () => reject(new Error("Google sign-in could not be loaded.")), { once: true });

      if (!existingScript) {
        script.id = googleScriptId;
        script.src = googleScriptUrl;
        script.async = true;
        script.defer = true;
        document.head.append(script);
      }
    }).catch((error) => {
      googleIdentityPromise = null;
      throw error;
    });
  }

  const identityPromise = googleIdentityPromise;

  if (!identityPromise) {
    throw new Error("Google sign-in is unavailable.");
  }

  return identityPromise;
}
