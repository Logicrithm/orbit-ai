# test_guard.py
from extract import _guard

fake = {
    "intent": "solver",                                   # invalid enum
    "planning": {"value": 85, "confidence": 75},          # clear, but...
    "evidence": {"planning": "felt relieved"},            # ...fabricated quote
    "energy": {"value": 85, "confidence": 75},            # clear, NO evidence at all
    "direction": "",                                      # empty noise
}
answer = "well i'm an extremely organized person, everyone says that about me"

result = _guard(fake, answer)
print(result)

assert "intent" not in result                 # enum guard
assert result["planning"]["confidence"] == 35 # fabricated quote -> downgraded
assert "planning" not in result.get("evidence", {})
assert result["energy"]["confidence"] == 35   # no evidence -> no certainty
assert "direction" not in result              # noise dropped
print("all guards hold")