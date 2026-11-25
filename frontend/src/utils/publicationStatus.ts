const labels: Record<string, string> = {
  verified_published: "Verified published record",
  verified_preprint: "Verified preprint record",
  registry_record: "Academic registry record found",
  conflicting: "Conflicting or ambiguous metadata",
  retracted: "Retracted or withdrawn",
  verification_unavailable: "Verification unavailable",
  pending: "Checking publication identity…",
  checking: "Checking publication identity…",
};

export function publicationStatus(status?: string) {
  return labels[status ?? ""] ?? "Identity not established";
}
