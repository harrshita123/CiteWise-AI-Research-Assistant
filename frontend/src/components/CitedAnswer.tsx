import type { AnswerSource, ExternalSource } from "../services/researchApi";

type Props = {
  text: string;
  sources: AnswerSource[];
  externalSources?: ExternalSource[];
  sourcePrefix: string;
  detailsId: string;
};

export default function CitedAnswer({ text, sources, externalSources = [], sourcePrefix, detailsId }: Props) {
  // Older saved answers may contain the model's marker and repeated links to one source.
  const displayText = text.replace(/(?:【(?:c[\w-]+|ext[\w-]+)】|\((?:c\d+-[0-9a-f]{8,}|ext[0-9a-f_-]{8,})\))/gi, "")
    .replace(/(\[(?:c[\w-]+|ext[\w-]+)\])(?:\s+\1)+/g, "$1")
    .replace(/\s+([.,;:!?])/g, "$1");
  const pieces = displayText.split(/(\[[^\]\n]+\])/g);
  return <>{pieces.map((piece, position) => {
    const id = piece.startsWith("[") && piece.endsWith("]") ? piece.slice(1, -1) : "";
    const external = externalSources.find((source) => source.id === id);
    if (external) return <a className="answer-citation" href={external.url} target="_blank" rel="noreferrer"
      key={position} aria-label={`Open external source ${external.title}`}>{piece}</a>;
    const pdfIndex = sources.findIndex((source) => source.id === id);
    if (pdfIndex >= 0) return <a className="answer-citation" href={`#${sourcePrefix}-${pdfIndex}`} key={position}
      aria-label={`Show PDF source on page ${sources[pdfIndex].page_number}`} onClick={() => {
        const details = document.getElementById(detailsId);
        if (details instanceof HTMLDetailsElement) details.open = true;
      }}>{piece}</a>;
    return piece;
  })}</>;
}
