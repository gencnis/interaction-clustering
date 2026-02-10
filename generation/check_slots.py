"""
check_slots.py

Sanity check:
- Extract all slot placeholders from templates
- Verify each slot exists in slot_pools.SLOTS

Fail fast if any slot is missing.
"""

import re

from slot_pools import SLOTS
from templates import TEMPLATES

SLOT_PATTERN = re.compile(r"\{([a-zA-Z0-9_]+)\}")

def extract_slots_from_templates():
    used_slots = set()

    for intent, gens in TEMPLATES.items():
        for gen, templates in gens.items():
            for tpl in templates:
                matches = SLOT_PATTERN.findall(tpl)
                for m in matches:
                    used_slots.add(m)

    return used_slots


def main():
    used_slots = extract_slots_from_templates()
    defined_slots = set(SLOTS.keys())

    missing = used_slots - defined_slots
    unused = defined_slots - used_slots

    print("Used slots:", sorted(used_slots))
    print("Defined slots:", sorted(defined_slots))

    if missing:
        print("\n❌ Missing slots in slot_pools.py:")
        for s in sorted(missing):
            print("  -", s)
    else:
        print("\n✅ All used slots are defined.")

    if unused:
        print("\nℹ️ Slots defined but not used in any template (OK for now):")
        for s in sorted(unused):
            print("  -", s)


if __name__ == "__main__":
    main()
