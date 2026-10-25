"""Per-test comparison of two pytest junit XML files: tests that fail or error on the branch but not on main, and
outcome counts for the tests whose id contains a filter string.
usage: junit_compare.py <main.xml> <branch.xml> [filter ...]"""
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def outcomes(path):
    result = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        test_id = f"{case.get('classname')}::{case.get('name')}"
        kinds = {child.tag for child in case}
        result[test_id] = next((k for k in ("error", "failure", "skipped") if k in kinds), "passed")
    return result


main, branch = outcomes(sys.argv[1]), outcomes(sys.argv[2])
bad = ("error", "failure")
only_branch = sorted(t for t, o in branch.items() if o in bad and main.get(t) not in bad)
print(f"main {dict(Counter(main.values()))}; branch {dict(Counter(branch.values()))}")
print(f"failing or erroring on the branch only: {len(only_branch)}", *only_branch[:20], sep="\n  ")
print(f"failing on both: {sorted(t for t, o in branch.items() if o in bad and main.get(t) in bad)}")
print(f"tests new on the branch: {len(set(branch) - set(main))}; tests gone from the branch: {sorted(set(main) - set(branch))[:10]}")
for needle in sys.argv[3:]:
    hits = Counter(o for t, o in branch.items() if needle in t)
    print(f"branch tests matching {needle!r}: {dict(hits)}")
