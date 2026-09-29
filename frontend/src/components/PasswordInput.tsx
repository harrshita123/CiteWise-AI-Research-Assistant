import { useState } from "react";

type Props = {
  id: string;
  name: string;
  label: string;
  autoComplete: "current-password" | "new-password";
};

export default function PasswordInput({ id, name, label, autoComplete }: Props) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <div className="password-input">
        <input
          id={id}
          name={name}
          type={isVisible ? "text" : "password"}
          autoComplete={autoComplete}
          minLength={8}
          maxLength={128}
          required
        />
        <button
          type="button"
          aria-label={`${isVisible ? "Hide" : "Show"} ${label.toLowerCase()}`}
          onClick={() => setIsVisible((visible) => !visible)}
        >
          {isVisible ? "Hide" : "Show"}
        </button>
      </div>
    </div>
  );
}
