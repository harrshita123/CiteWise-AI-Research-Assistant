interface GoogleCredentialResponse {
  credential: string;
  select_by?: string;
}

interface GoogleIdentityApi {
  initialize: (configuration: {
    client_id: string;
    callback: (response: GoogleCredentialResponse) => void;
  }) => void;
  renderButton: (
    parent: HTMLElement,
    options: {
      theme?: "outline" | "filled_black" | "filled_blue";
      size?: "large" | "medium" | "small";
      text?: "signin_with" | "signup_with" | "continue_with" | "signin";
      shape?: "rectangular" | "pill" | "circle" | "square";
      width?: number;
    },
  ) => void;
}

interface Window {
  google?: {
    accounts: {
      id: GoogleIdentityApi;
    };
  };
}
