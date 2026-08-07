"""Offline tests for the tool layer and the rules baked into it.

The point of these is that the rules live in the functions, not the prompt.
A prompt can be talked into cancelling a delivered order; a function that
refuses cannot. Each refusal also has to explain itself, so the agent has
something to say beyond "I can't do that".

Run: python test_tools.py
"""
from __future__ import annotations

from tools import (
    ORDER_TOOLS,
    HANDLERS,
    execute,
    get_handoffs,
    get_orders,
    get_tickets,
    reset_store,
)

failures: list[str] = []


def check(label: str, ok: bool, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f" -> {detail}"))
    if not ok:
        failures.append(label)


reset_store()

print("Schemas:")
names = [t["name"] for t in ORDER_TOOLS]
check("every schema has a handler", sorted(names) == sorted(HANDLERS), names)
check("no duplicate tool names", len(names) == len(set(names)), names)
for tool in ORDER_TOOLS:
    params = tool["parameters"]
    ok = tool.get("description") and params.get("type") == "object" and "properties" in params
    if not ok:
        check(f"{tool['name']} is well formed", False, tool)
check("all schemas well formed", True)

print("\nLooking an order up:")
found = execute("check_order_status", {"order_number": "ord-1042"})
check("case does not matter", found.get("status") == "out_for_delivery", found)
missing = execute("check_order_status", {"order_number": "ORD-9999"})
check("a miss says what to do next", missing.get("error") == "not_found" and "hint" in missing,
      missing)

print("\nFinding orders without a number:")
by_email = execute("find_orders", {"email": "DANA@example.com"})
check("email match ignores case", by_email.get("count") == 2, by_email)
by_phone = execute("find_orders", {"phone": "555-0199"})
check("phone match works on the last digits",
      by_phone.get("count") == 1
      and by_phone["orders"][0]["order_number"] == "ORD-3377", by_phone)
check("no search term is an error, not a crash",
      execute("find_orders", {}).get("error") == "missing_search_term")
check("an unknown caller gets a hint",
      execute("find_orders", {"email": "nobody@example.com"}).get("error") == "no_orders_found")

print("\nCancelling, and refusing to:")
late = execute("cancel_order", {"order_number": "ORD-1042"})
check("a shipped order cannot be cancelled", late.get("error") == "not_cancellable", late)
check("the refusal explains itself", "hint" in late and late.get("order_status"), late)
check("a shipped order is left alone",
      get_orders()["ORD-1042"]["status"] == "out_for_delivery")
delivered = execute("cancel_order", {"order_number": "ORD-3377"})
check("a delivered order cannot be cancelled", delivered.get("error") == "not_cancellable")
done = execute("cancel_order", {"order_number": "ORD-2210"})
check("a processing order cancels", done.get("status") == "cancelled", done)
check("cancelling states the refund", done.get("refund_usd") == 129.00, done)
check("the store really changed", get_orders()["ORD-2210"]["status"] == "cancelled")

print("\nDelivery instructions:")
reset_store()
ok = execute("update_delivery_instructions",
             {"order_number": "ORD-1042", "instructions": "Leave with the doorman"})
check("in-flight orders can be redirected", ok.get("status") == "updated", ok)
check("the record shows it",
      get_orders()["ORD-1042"]["delivery_instructions"] == "Leave with the doorman")
too_late = execute("update_delivery_instructions",
                   {"order_number": "ORD-3377", "instructions": "Leave out back"})
check("a delivered order cannot be redirected",
      too_late.get("error") == "already_delivered", too_late)

print("\nTickets:")
ticket = execute("create_support_ticket",
                 {"order_number": "ORD-2210", "reason": "Never arrived"})
check("a ticket comes back with a reference", ticket.get("ticket_id", "").startswith("TCK-"),
      ticket)
check("the ticket warns against promising a time", "note" in ticket, ticket)
status = execute("check_ticket_status", {"ticket_id": ticket["ticket_id"].lower()})
check("a ticket can be looked up again", status.get("status") == "open", status)
check("an unknown reference is handled",
      execute("check_ticket_status", {"ticket_id": "TCK-9999"}).get("error") == "not_found")
check("the store lists it", len(get_tickets()) == 1)

print("\nHanding over to a person:")
handoff = execute("transfer_to_human",
                  {"reason": "Double charge on the card", "order_number": "ORD-1042"})
check("the transfer is acknowledged", handoff.get("status") == "transferring", handoff)
check("the handover is recorded", len(get_handoffs()) == 1, get_handoffs())

print("\nSafety:")
check("private contact details never reach the model",
      all("email" not in o and "phone" not in o for o in get_orders().values()),
      get_orders())
check("an unknown tool is an error, not a crash",
      execute("nope", {}).get("error", "").startswith("unknown_tool"))
check("bad arguments are an error, not a crash",
      execute("check_order_status", {"wrong": "x"}).get("error", "").startswith("bad_arguments"))

reset_store()
check("reset restores the demo data",
      get_orders()["ORD-1042"]["delivery_instructions"] == "Leave at front door"
      and get_orders()["ORD-2210"]["status"] == "processing"
      and get_tickets() == [] and get_handoffs() == [])

print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    raise SystemExit(1)
print("All tool checks passed.")
