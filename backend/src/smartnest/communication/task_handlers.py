"""Task handler implementations for the Remote Agent Server (Agent B).

Each handler is a callable that receives an incoming ``MessageEnvelope``
and returns an optional response ``MessageEnvelope``.

Handlers are registered with the ``MessageRouter`` so the server can
dispatch incoming messages without knowing the domain logic.

Assignment 2.2 handlers:
    - ``handle_state_sync`` — validates and stores apartment state.
    - ``handle_task_request`` — dispatches to registered task evaluators.

Task evaluators (e.g. ``evaluate_furniture_fit``) are registered
separately so new evaluation tasks can be added without modifying the
handler logic.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

from smartnest.core.models import MessageEnvelope, MessageType

logger = logging.getLogger(__name__)

# Type alias for a task evaluator function.
TaskEvaluator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class TaskHandlerRegistry:
    """Registry of task evaluators and message handlers for Agent B.

    Holds:
        - The synchronized apartment state received from Agent A.
        - Registered task evaluators keyed by ``task_type``.
        - Message handlers for STATE_SYNC and TASK_REQUEST.

    The handlers are returned as bound methods so they can be directly
    registered with a ``MessageRouter``.
    """

    def __init__(self, agent_id: str = "agent_b") -> None:
        self._agent_id = agent_id
        self._apartment_state: dict[str, Any] = {}
        self._evaluators: dict[str, TaskEvaluator] = {}

    # ── State access ─────────────────────────────────────────────────────

    @property
    def apartment_state(self) -> dict[str, Any]:
        """Return a deep copy of the current apartment state."""
        return copy.deepcopy(self._apartment_state)

    def has_state(self) -> bool:
        """Whether apartment state has been synchronized."""
        return bool(self._apartment_state)

    # ── Task evaluator registration ──────────────────────────────────────

    def register_evaluator(self, task_type: str, evaluator: TaskEvaluator) -> None:
        """Register an evaluator function for a task type.

        Args:
            task_type: String identifier (e.g. "evaluate_furniture_fit").
            evaluator: Callable(apartment_state, task_payload) -> result dict.
        """
        self._evaluators[task_type] = evaluator
        logger.debug("Registered evaluator for task_type=%s", task_type)

    @property
    def registered_task_types(self) -> list[str]:
        """Return registered task type strings."""
        return list(self._evaluators.keys())

    # ── Message handlers ────────────────────────────────────────────────

    def handle_state_sync(self, message: MessageEnvelope) -> MessageEnvelope:
        """Validate and store the incoming apartment state.

        Returns an ACK message to the sender.
        """
        payload = message.payload or {}

        # Basic validation — must contain project_id
        project_id = payload.get("project_id")
        if not project_id:
            logger.warning("STATE_SYNC missing project_id from %s", message.sender)
            return MessageEnvelope(
                sender=self._agent_id,
                receiver=message.sender,
                message_type=MessageType.FAILURE,
                task_id=message.task_id,
                payload={"reason": "missing_project_id"},
            )

        rooms_error = _validate_rooms(payload.get("rooms", {}))
        if rooms_error:
            logger.warning("STATE_SYNC with invalid rooms from %s: %s", message.sender, rooms_error)
            return MessageEnvelope(
                sender=self._agent_id,
                receiver=message.sender,
                message_type=MessageType.FAILURE,
                task_id=message.task_id,
                payload={"reason": "invalid_rooms", "error": rooms_error},
            )

        self._apartment_state = dict(payload)
        logger.info(
            "State synchronized: project_id=%s, rooms=%s, budget=%s",
            project_id,
            list(payload.get("rooms", {}).keys()),
            payload.get("budget"),
        )

        return MessageEnvelope(
            sender=self._agent_id,
            receiver=message.sender,
            message_type=MessageType.ACK,
            task_id=message.task_id,
            payload={
                "status": "synchronized",
                "project_id": project_id,
            },
        )

    def handle_task_request(self, message: MessageEnvelope) -> MessageEnvelope:
        """Dispatch a task request to the registered evaluator.

        Returns an INFORM_RESULT message containing the evaluation result.
        """
        payload = message.payload or {}
        task_type = payload.get("task_type")

        if not task_type:
            return MessageEnvelope(
                sender=self._agent_id,
                receiver=message.sender,
                message_type=MessageType.FAILURE,
                task_id=message.task_id,
                payload={"reason": "missing_task_type"},
            )

        evaluator = self._evaluators.get(task_type)
        if evaluator is None:
            return MessageEnvelope(
                sender=self._agent_id,
                receiver=message.sender,
                message_type=MessageType.FAILURE,
                task_id=message.task_id,
                payload={
                    "reason": "unknown_task_type",
                    "task_type": task_type,
                    "available": self.registered_task_types,
                },
            )

        if not self.has_state():
            return MessageEnvelope(
                sender=self._agent_id,
                receiver=message.sender,
                message_type=MessageType.FAILURE,
                task_id=message.task_id,
                payload={"reason": "no_synchronized_state"},
            )

        try:
            # Evaluators get a copy so they can't mutate the stored state.
            result = evaluator(self.apartment_state, payload)
        except Exception as exc:
            logger.exception("Evaluator %s raised: %s", task_type, exc)
            return MessageEnvelope(
                sender=self._agent_id,
                receiver=message.sender,
                message_type=MessageType.FAILURE,
                task_id=message.task_id,
                payload={
                    "reason": "evaluator_exception",
                    "error": str(exc),
                },
            )

        logger.info("Task %s completed: %s", task_type, result)

        return MessageEnvelope(
            sender=self._agent_id,
            receiver=message.sender,
            message_type=MessageType.INFORM_RESULT,
            task_id=message.task_id,
            payload={
                "task_type": task_type,
                "result": result,
            },
        )


# ── State validation ─────────────────────────────────────────────────────

def _validate_rooms(rooms: Any) -> str:
    """Return an error description if ``rooms`` is malformed, else ``""``.

    ``rooms`` must be a mapping of room name → dict with a numeric,
    non-negative ``width_cm``; ``length_cm`` is optional but must be
    numeric and non-negative if given.
    """
    if not isinstance(rooms, dict):
        return (
            "'rooms' must be an object mapping room names to dimensions, "
            f"got {type(rooms).__name__}"
        )
    for name, room in rooms.items():
        if not isinstance(room, dict):
            return f"room '{name}' must be an object"
        if "width_cm" not in room:
            return f"room '{name}' is missing width_cm"
        for key in ("width_cm", "length_cm"):
            if key not in room:
                continue
            value = room[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                return f"room '{name}' has invalid {key}: {value!r}"
    return ""


# ── Built-in evaluator: furniture fit ────────────────────────────────────

def evaluate_furniture_fit(
    apartment_state: dict[str, Any],
    task_payload: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate whether a piece of furniture fits in a room.

    Uses the synchronized apartment state to check room dimensions and
    remaining budget.  The output depends entirely on the received state —
    nothing is hardcoded.

    Expected ``task_payload`` keys:
        - ``product``: dict with ``product_id``, ``width_cm``, ``price``,
          and optionally ``depth_cm`` and ``target_room``.

    Expected ``apartment_state`` keys:
        - ``rooms``: dict of room_name → {width_cm, length_cm}.
        - ``remaining_budget``: numeric.

    Room selection:
        - ``target_room`` given and known → that room.
        - ``target_room`` given but unknown → not evaluated; the result
          reports ``error: "unknown_target_room"`` (it never silently falls
          back to a different room).
        - no ``target_room`` → the first room in the state, reported in
          ``room_name`` so the caller can see which room was used.

    Dimension checks: product width vs room width; if the product gives
    ``depth_cm`` and the room gives ``length_cm``, depth vs length too.
    """
    product = task_payload.get("product", {})
    rooms = apartment_state.get("rooms", {})
    remaining_budget = apartment_state.get("remaining_budget", 0)
    product_id = product.get("product_id", "product")

    product_width = product.get("width_cm", 0)
    product_depth = product.get("depth_cm")
    product_price = product.get("price", 0)
    within_budget = product_price <= remaining_budget

    # Determine which room to evaluate against
    target_room_name = product.get("target_room")
    if target_room_name:
        if target_room_name not in rooms:
            return {
                "fits": False,
                "error": "unknown_target_room",
                "room_name": target_room_name,
                "available_rooms": list(rooms.keys()),
                "within_budget": within_budget,
                "remaining_budget": remaining_budget,
                "product_price": product_price,
                "recommendation": (
                    f"Cannot evaluate {product_id}: room '{target_room_name}' "
                    f"is not in the apartment state."
                ),
            }
        room_name = target_room_name
    elif rooms:
        room_name = next(iter(rooms))
    else:
        return {
            "fits": False,
            "width_excess_cm": 0,
            "within_budget": within_budget,
            "recommendation": "No rooms available in apartment state.",
        }

    room = rooms[room_name]
    room_width = room.get("width_cm", 0)
    room_length = room.get("length_cm")
    width_excess = product_width - room_width
    fits_width = width_excess <= 0

    depth_excess = None
    fits_depth = True
    if product_depth is not None and room_length is not None:
        depth_excess = product_depth - room_length
        fits_depth = depth_excess <= 0

    fits = fits_width and fits_depth

    if not fits_width:
        recommendation = (
            f"Reject {product_id} because it exceeds available room width "
            f"by {abs(width_excess)} cm."
        )
    elif not fits_depth:
        recommendation = (
            f"Reject {product_id} because its depth exceeds the room length "
            f"by {abs(depth_excess)} cm."
        )
    elif not within_budget:
        recommendation = (
            f"Reject {product_id} because its price exceeds the remaining budget."
        )
    else:
        recommendation = (
            f"Accept {product_id}: fits {room_name} "
            f"({product_width} cm ≤ {room_width} cm) and within budget."
        )

    result = {
        "fits": fits,
        "room_name": room_name,
        "width_excess_cm": width_excess,
        "room_width_cm": room_width,
        "product_width_cm": product_width,
        "within_budget": within_budget,
        "remaining_budget": remaining_budget,
        "product_price": product_price,
        "recommendation": recommendation,
    }
    if depth_excess is not None:
        result["depth_excess_cm"] = depth_excess
        result["room_length_cm"] = room_length
        result["product_depth_cm"] = product_depth
    return result
