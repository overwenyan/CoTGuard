import sys
sys.path.insert(0, 'experiments/relay')
from score_repetition_coded import anchor_text_of, mid_fragment

tests = [
    "Work through this, and work backwards from the quantity being asked for.",
    "Approach this carefully and compute every intermediate quantity twice, by two different routes.",
    "Take care to determine what is unknown before performing any arithmetic.",
    "As you reason, start from whichever quantity has the fewest dependencies.",
]
for t in tests:
    print(repr(anchor_text_of(t)))

text = "Step1\nStep2\nStep3\nStep4\nStep5"
print("5 steps, skip1/1 ->", repr(mid_fragment(text)))
text2 = "Step1\nStep2\nStep3"
print("3 steps, skip1/1 -> (should be empty)", repr(mid_fragment(text2)))
