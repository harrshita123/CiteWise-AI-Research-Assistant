export function deadlineFromRetry(seconds: number | null, now?: number): number;
export function remainingSeconds(deadline: number, now?: number): number;
export function formatRetryDuration(seconds: number): string;
