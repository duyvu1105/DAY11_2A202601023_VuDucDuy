"""
Lab 11 — Part 4: Human-in-the-Loop Design
  TODO 11: Confidence Router
  TODO 12: Design 3 HITL decision points
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


# ============================================================
# TODO 11: Implement ConfidenceRouter
#
# Route agent responses based on confidence scores:
#   - HIGH (>= 0.9): Auto-send to user
#   - MEDIUM (0.7 - 0.9): Queue for human review
#   - LOW (< 0.7): Escalate to human immediately
#
# Special case: if the action is HIGH_RISK (e.g., money transfer,
# account deletion), ALWAYS escalate regardless of confidence.
#
# Implement the route() method.
# ============================================================

HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Thresholds:
        HIGH:   confidence >= 0.9 -> auto-send
        MEDIUM: 0.7 <= confidence < 0.9 -> queue for review
        LOW:    confidence < 0.7 -> escalate to human

    High-risk actions always escalate regardless of confidence.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        # 1. High-risk actions ALWAYS go to a human, even at 0.99 confidence.
        if action_type in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {action_type}",
                priority="high",
                requires_human=True,
            )

        # 2. Confidence thresholds for ordinary banking answers.
        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )
        if confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence — needs review",
                priority="normal",
                requires_human=True,
            )
        return RoutingDecision(
            action="escalate",
            confidence=confidence,
            reason="Low confidence — escalating",
            priority="high",
            requires_human=True,
        )


# ============================================================
# Review lifecycle: approve / reject / timeout + audit trail
# ============================================================


@dataclass
class ReviewTask:
    """One HITL decision with a full audit trail."""

    request_id: str
    user_id: str
    intent: str
    proposed_action: str
    context: str  # what the reviewer must see (question, draft, diff, risk)
    confidence: float
    status: str = "pending"  # pending | approved | rejected | timeout
    reviewer_id: str | None = None
    audit: list[dict] = field(default_factory=list)
    approval_id: str | None = None

    def _append(self, event: str, **extra) -> None:
        self.audit.append(
            {
                "event": event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **extra,
            }
        )

    def approve(self, reviewer_id: str) -> dict:
        self.status = "approved"
        self.reviewer_id = reviewer_id
        self.approval_id = f"HITL-{uuid4().hex[:8].upper()}"
        self._append("approved", reviewer_id=reviewer_id, approval_id=self.approval_id)
        return self.audit[-1]

    def reject(self, reviewer_id: str, reason: str) -> dict:
        self.status = "rejected"
        self.reviewer_id = reviewer_id
        self._append("rejected", reviewer_id=reviewer_id, reason=reason)
        return self.audit[-1]

    def timeout(self, timeout_seconds: int = 86400) -> dict:
        self.status = "timeout"
        self._append(
            "timeout",
            timeout_seconds=timeout_seconds,
            fallback="action not executed",
        )
        return self.audit[-1]


class HITLReviewWorkflow:
    """Create review tasks and record approve/reject/timeout decisions."""

    def __init__(self):
        self.tasks: dict[str, ReviewTask] = {}

    def create(
        self,
        *,
        user_id: str,
        intent: str,
        proposed_action: str,
        context: str,
        confidence: float,
        request_id: str | None = None,
    ) -> ReviewTask:
        request_id = request_id or f"REQ-{uuid4().hex[:12].upper()}"
        task = ReviewTask(
            request_id=request_id,
            user_id=user_id,
            intent=intent,
            proposed_action=proposed_action,
            context=context,
            confidence=confidence,
        )
        task._append("created", routing="escalate")
        self.tasks[request_id] = task
        return task

    def decide(
        self,
        request_id: str,
        decision: str,
        reviewer_id: str = "reviewer-1",
        reason: str | None = None,
        timeout_seconds: int = 86400,
    ) -> dict:
        """Approve / reject / timeout a pending review and return audit row."""
        task = self.tasks[request_id]
        if decision == "approve":
            return task.approve(reviewer_id)
        if decision == "reject":
            return task.reject(reviewer_id, reason or "not approved")
        if decision == "timeout":
            return task.timeout(timeout_seconds)
        raise ValueError(f"Unknown decision: {decision}")


# ============================================================
# TODO 12: Design 3 HITL decision points + a review lifecycle
#
# For each decision point, define:
# - trigger: What condition activates this HITL check?
# - hitl_model: Which model? (human-in-the-loop, human-on-the-loop,
#   human-as-tiebreaker)
# - context_needed: What info does the human reviewer need?
# - example: A concrete scenario
# - approval_path: What approve/reject/timeout decision is recorded?
# - audit_fields: Which correlation ID, intent and proposed action/diff are logged?
#
# Think about real banking scenarios where human judgment is critical.
# ============================================================

hitl_decision_points = [
    {
        "id": 1,
        "name": "Money transfer authorization",
        "trigger": (
            "ConfidenceRouter routes action_type='transfer_money' (always "
            "escalate) or any beneficiary/amount change for an existing transfer."
        ),
        "hitl_model": "human-in-the-loop (no side effect without recorded approval)",
        "context_needed": (
            "request_id; sender + recipient account masks; amount + fee; "
            "proposed transfer diff vs. customer request; fraud/sanction flags."
        ),
        "example": (
            "Customer asks to send 500,000,000 VND to a brand-new beneficiary. "
            "Agent proposes the transfer; reviewer sees the diff and must approve "
            "before egress to https://api.vinbank.example/v1/transfers."
        ),
        "approval_path": (
            "Approve → creates approval_id HITL-XXXXXXXX and executes; "
            "Reject → transfer cancelled with reason; Timeout (24h) → auto-reject "
            "and notify customer to re-initiate."
        ),
        "audit_fields": (
            "request_id, intent='transfer_money', proposed_action + payload diff, "
            "destination allowlist result, reviewer_id, decision, approval_id, timestamp."
        ),
    },
    {
        "id": 2,
        "name": "Account closure / sensitive profile change",
        "trigger": (
            "action_type in ('close_account','change_password','delete_data',"
            "'update_personal_info') regardless of confidence."
        ),
        "hitl_model": "human-as-tiebreaker (agent proposes, human confirms)",
        "context_needed": (
            "Account summary (balance, linked products), identity verification "
            "status, outstanding loan/card obligations, proposed change diff."
        ),
        "example": (
            "Customer requests account closure while a 200M VND personal loan is "
            "outstanding; human must confirm the settlement plan before closure."
        ),
        "approval_path": (
            "Approve → routes to back-office queue with approval_id; Reject → "
            "account kept, reason recorded; Timeout → fallback message asking the "
            "customer to visit a branch (no destructive action runs)."
        ),
        "audit_fields": (
            "request_id, intent, proposed_action diff, account summary hash, "
            "verification result, reviewer_id, decision, timeout fallback, timestamp."
        ),
    },
    {
        "id": 3,
        "name": "Low-confidence / disputed rate answer",
        "trigger": (
            "confidence < 0.7 for general answers, or judge flags a factual "
            "dispute against ground truth (e.g. savings rate mismatch)."
        ),
        "hitl_model": "human-on-the-loop (reviewer watches, approves corrections)",
        "context_needed": (
            "Customer question, model draft answer, VinBank ground-truth rate, "
            "and the diff between draft and correct value."
        ),
        "example": (
            "Customer insists the 12-month savings rate is 5.5%; model draft says "
            "4.25% but with low confidence — reviewer approves the corrected reply."
        ),
        "approval_path": (
            "Approve → corrected answer sent; Reject → generic 'please verify at "
            "branch/hotline' fallback; Timeout → send the safe generic fallback "
            "and log the pending correction for follow-up."
        ),
        "audit_fields": (
            "request_id, question, draft vs. corrected diff, ground_truth_ref, "
            "judge verdict, reviewer_id, decision, timeout behavior, timestamp."
        ),
    },
]


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    Example:  {point['example']}")
        print(f"    Approval: {point['approval_path']}")
        print(f"    Audit:    {point['audit_fields']}")
    print("\n" + "=" * 60)


def test_review_lifecycle():
    """Demonstrate approve / reject / timeout with a correlated audit trail."""
    workflow = HITLReviewWorkflow()
    task = workflow.create(
        user_id="customer-42",
        intent="transfer_money",
        proposed_action="transfer 500000000 VND to new beneficiary",
        context="New beneficiary; amount above 100M VND threshold.",
        confidence=0.98,
        request_id="REQ-ABC123",
    )
    workflow.decide("REQ-ABC123", "approve", reviewer_id="reviewer-1")
    timeout_task = workflow.create(
        user_id="customer-43",
        intent="update_personal_info",
        proposed_action="change phone number",
        context="Identity verification pending.",
        confidence=0.65,
        request_id="REQ-DEF456",
    )
    workflow.decide("REQ-DEF456", "timeout")

    print("\nReview lifecycle (approve):")
    print("  request_id:", task.request_id, "| status:", task.status)
    print("  approval_id:", task.approval_id, "| reviewer:", task.reviewer_id)
    for row in task.audit:
        print("  audit:", row)
    print("\nReview lifecycle (timeout):")
    print("  request_id:", timeout_task.request_id, "| status:", timeout_task.status)
    for row in timeout_task.audit:
        print("  audit:", row)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.config import ensure_utf8_stdio

    ensure_utf8_stdio()
    test_confidence_router()
    test_hitl_points()
    test_review_lifecycle()
