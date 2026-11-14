export function matchesAnswerRequest(answer, request, activeRequestId, contextVersion) {
  return activeRequestId === request.requestId && contextVersion === request.contextVersion
    && answer.request_id === request.requestId && answer.question === request.question
    && answer.paper_id === request.paperId
    && (!request.conversationId || answer.conversation_id === request.conversationId);
}
