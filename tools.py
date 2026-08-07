"""Order tools for the customer-support voice agent.

Each tool is a JSON schema (for session.tools) plus a plain Python function
that runs on our side. The model never touches the order database directly;
it only ever sees what these functions return.

The rules live here rather than in the prompt. A prompt can be talked out of
refusing to cancel a delivered order; a function cannot. When a tool refuses,
it says why and what to do instead, so the agent can explain the policy
instead of apologising vaguely or inventing one.
"""

# In-memory order store standing in for a real database.
_ORDERS = {
    "ORD-1042": {
        "status": "out_for_delivery",
        "eta": "2026-08-05",
        "delivery_instructions": "Leave at front door",
        "email": "dana@example.com",
        "phone": "+1-555-0142",
        "total_usd": 84.50,
    },
    "ORD-2210": {
        "status": "processing",
        "eta": "2026-08-09",
        "delivery_instructions": None,
        "email": "dana@example.com",
        "phone": "+1-555-0142",
        "total_usd": 129.00,
    },
    "ORD-3377": {
        "status": "delivered",
        "eta": "2026-07-30",
        "delivery_instructions": "Left with neighbour",
        "email": "sam@example.com",
        "phone": "+1-555-0199",
        "total_usd": 42.00,
    },
}

_TICKETS = []
_HANDOFFS = []

# Fields the caller never needs read back, and that a voice agent should not
# be reciting out loud.
_PRIVATE = ("email", "phone")

# Once an order is on a van or through the door, changing it is no longer a
# database edit, it is a logistics problem for a human.
_LOCKED_FOR_CHANGES = ("out_for_delivery", "delivered")
_CANCELLABLE = ("processing", "packing")


def _public(order_number: str, order: dict) -> dict:
    return {
        "order_number": order_number,
        **{k: v for k, v in order.items() if k not in _PRIVATE},
    }


def check_order_status(order_number: str) -> dict:
    order = _ORDERS.get(order_number.upper())
    if order is None:
        return {"error": "not_found", "order_number": order_number,
                "hint": "Offer to look the order up by email or phone instead."}
    return _public(order_number.upper(), order)


def find_orders(email: str = "", phone: str = "") -> dict:
    """Look up orders when the caller does not have the number to hand."""
    needle_email = (email or "").strip().lower()
    needle_phone = "".join(ch for ch in (phone or "") if ch.isdigit())
    if not needle_email and not needle_phone:
        return {"error": "missing_search_term",
                "hint": "Ask the caller for the email or phone on the order."}

    matches = []
    for number, order in _ORDERS.items():
        by_email = needle_email and order["email"].lower() == needle_email
        digits = "".join(ch for ch in order["phone"] if ch.isdigit())
        by_phone = needle_phone and digits.endswith(needle_phone[-7:])
        if by_email or by_phone:
            matches.append({"order_number": number, "status": order["status"],
                            "eta": order["eta"]})
    if not matches:
        return {"error": "no_orders_found",
                "hint": "Nothing on that email or phone. Offer a support ticket."}
    return {"orders": matches, "count": len(matches)}


def update_delivery_instructions(order_number: str, instructions: str) -> dict:
    number = order_number.upper()
    order = _ORDERS.get(number)
    if order is None:
        return {"error": "not_found", "order_number": order_number}
    if order["status"] == "delivered":
        return {"error": "already_delivered", "order_number": number,
                "hint": "It is already delivered, so instructions cannot change. "
                        "Offer a support ticket if it went to the wrong place."}
    order["delivery_instructions"] = instructions
    return {"order_number": number, "status": "updated",
            "delivery_instructions": instructions}


def cancel_order(order_number: str) -> dict:
    """Cancel, but only while cancelling still means anything."""
    number = order_number.upper()
    order = _ORDERS.get(number)
    if order is None:
        return {"error": "not_found", "order_number": order_number}
    if order["status"] not in _CANCELLABLE:
        return {
            "error": "not_cancellable",
            "order_number": number,
            "order_status": order["status"],
            "hint": ("Too late to cancel once it has left the warehouse. Offer a "
                     "return once it arrives, or a support ticket to chase it."),
        }
    order["status"] = "cancelled"
    order["eta"] = None
    return {"order_number": number, "status": "cancelled",
            "refund_usd": order["total_usd"],
            "note": "Refund goes back to the original payment method."}


def create_support_ticket(order_number: str, reason: str) -> dict:
    ticket_id = f"TCK-{len(_TICKETS) + 1000}"
    _TICKETS.append({"ticket_id": ticket_id, "order_number": order_number.upper(),
                     "reason": reason, "status": "open"})
    return {"ticket_id": ticket_id, "status": "created",
            "note": "The team replies by email or phone. Do not promise a time."}


def check_ticket_status(ticket_id: str) -> dict:
    for ticket in _TICKETS:
        if ticket["ticket_id"].upper() == ticket_id.upper():
            return dict(ticket)
    return {"error": "not_found", "ticket_id": ticket_id,
            "hint": "No ticket with that reference. Offer to open a new one."}


def transfer_to_human(reason: str, order_number: str = "") -> dict:
    """Hand the call to a person and stop talking."""
    _HANDOFFS.append({"reason": reason, "order_number": order_number.upper()})
    return {"status": "transferring", "reason": reason,
            "note": "Tell the caller you are transferring them, then stop."}


def get_orders() -> dict:
    """Read the live order store. The demo UI shows this next to the
    transcript so you can check what the agent actually changed instead of
    trusting what it said it changed."""
    return {number: _public(number, order) for number, order in _ORDERS.items()}


def get_tickets() -> list:
    return list(_TICKETS)


def get_handoffs() -> list:
    return list(_HANDOFFS)


def reset_store():
    """Put the demo data back after a run that changed an order."""
    _ORDERS["ORD-1042"].update({
        "status": "out_for_delivery", "eta": "2026-08-05",
        "delivery_instructions": "Leave at front door",
    })
    _ORDERS["ORD-2210"].update({
        "status": "processing", "eta": "2026-08-09",
        "delivery_instructions": None,
    })
    _ORDERS["ORD-3377"].update({
        "status": "delivered", "eta": "2026-07-30",
        "delivery_instructions": "Left with neighbour",
    })
    _TICKETS.clear()
    _HANDOFFS.clear()


def _order_arg(description="Order number, e.g. ORD-1042"):
    return {"type": "string", "description": description}


# Tool schemas sent in session.update -> session.tools. Names, descriptions
# and required fields matter: they are what the model reads to decide when
# and how to call each function.
ORDER_TOOLS = [
    {
        "type": "function",
        "name": "check_order_status",
        "description": "Look up the current status, ETA, and delivery instructions for an order.",
        "parameters": {
            "type": "object",
            "properties": {"order_number": _order_arg()},
            "required": ["order_number"],
        },
    },
    {
        "type": "function",
        "name": "find_orders",
        "description": (
            "Find a caller's orders from the email or phone number on the account. "
            "Use this when the caller does not know their order number."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Email on the order"},
                "phone": {"type": "string", "description": "Phone number on the order"},
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "update_delivery_instructions",
        "description": (
            "Change the delivery instructions for an order. Only call this after the "
            "caller has explicitly confirmed the new instructions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_number": _order_arg(),
                "instructions": {"type": "string", "description": "New delivery instructions"},
            },
            "required": ["order_number", "instructions"],
        },
    },
    {
        "type": "function",
        "name": "cancel_order",
        "description": (
            "Cancel an order and start a refund. Only works before the order ships. "
            "Always read the order number and the refund back to the caller and get "
            "an explicit yes before calling this."
        ),
        "parameters": {
            "type": "object",
            "properties": {"order_number": _order_arg()},
            "required": ["order_number"],
        },
    },
    {
        "type": "function",
        "name": "create_support_ticket",
        "description": "Open a support ticket when an issue cannot be resolved on the call.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_number": _order_arg("Order number the ticket relates to"),
                "reason": {"type": "string", "description": "Short summary of the issue"},
            },
            "required": ["order_number", "reason"],
        },
    },
    {
        "type": "function",
        "name": "check_ticket_status",
        "description": "Look up a support ticket the caller already has, by its reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "Ticket reference, e.g. TCK-1000"},
            },
            "required": ["ticket_id"],
        },
    },
    {
        "type": "function",
        "name": "transfer_to_human",
        "description": (
            "Hand the call to a human agent. Use this for refunds outside a plain "
            "cancellation, payment or billing disputes, account access, anything you "
            "are unsure about, or a caller who is upset or asks for a person."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why the call needs a person"},
                "order_number": _order_arg("Related order number, if there is one"),
            },
            "required": ["reason"],
        },
    },
]

HANDLERS = {
    "check_order_status": check_order_status,
    "find_orders": find_orders,
    "update_delivery_instructions": update_delivery_instructions,
    "cancel_order": cancel_order,
    "create_support_ticket": create_support_ticket,
    "check_ticket_status": check_ticket_status,
    "transfer_to_human": transfer_to_human,
}


def execute(name: str, arguments: dict) -> dict:
    """Run a tool by name. Never raises: failures come back as an error dict
    so the model can speak them instead of the connection breaking."""
    handler = HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown_tool: {name}"}
    try:
        return handler(**arguments)
    except TypeError as exc:
        return {"error": f"bad_arguments: {exc}"}
