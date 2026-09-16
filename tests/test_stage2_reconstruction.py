"""
Unit tests for Stage 2: Conversation Reconstruction logic.
Covers core graph traversal, root discovery, cycle detection, branching, and edge cases.
"""

import unittest
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set


def normalize_relationship_ids(val: Any) -> Optional[int]:
    """Normalize float/string/int IDs to int or None."""
    if val is None or val == "" or str(val).lower() == "nan":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def parse_response_ids(val: Any) -> List[int]:
    """Parse comma-separated or single response tweet IDs."""
    if val is None or val == "" or str(val).lower() == "nan":
        return []
    result = []
    for item in str(val).split(","):
        item = item.strip()
        if item and item.lower() != "nan":
            try:
                result.append(int(float(item)))
            except (ValueError, TypeError):
                pass
    return result


def reconstruct_conversation_graph(
    tweets: List[Dict[str, Any]]
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Tuple[int, List[int]]], Dict[str, int]]:
    """
    Modular reconstruction logic for testing.
    Returns:
      - threads: dict mapping root_id to ordered list of turn dicts
      - cycles: list of detected cycle paths
      - stats: dict of missing reference counts and branching stats
    """
    tweet_lookup = {}
    parent_map = {}
    children_map = defaultdict(list)

    for tw in tweets:
        tid = int(tw["tweet_id"])
        pid = normalize_relationship_ids(tw.get("in_response_to_tweet_id"))
        raw_resp = tw.get("response_tweet_id")
        resp_ids = parse_response_ids(raw_resp)

        inbound = bool(tw.get("inbound", True))
        speaker = "customer" if inbound else "AppleSupport"

        tweet_lookup[tid] = {
            "tweet_id": tid,
            "author_id": str(tw.get("author_id", "user")),
            "speaker": speaker,
            "inbound": inbound,
            "created_at": tw.get("created_at", ""),
            "text": str(tw.get("text", "")),
            "in_response_to_tweet_id": pid,
            "response_tweet_id": str(raw_resp) if raw_resp is not None else None,
            "parsed_response_ids": resp_ids,
        }
        parent_map[tid] = pid
        if pid is not None:
            children_map[pid].append(tid)

    # Missing references check
    missing_parents_count = 0
    for tid, pid in parent_map.items():
        if pid is not None and pid not in tweet_lookup:
            missing_parents_count += 1

    missing_children_count = 0
    for tid, tw_data in tweet_lookup.items():
        for cid in tw_data["parsed_response_ids"]:
            if cid not in tweet_lookup:
                missing_children_count += 1

    # Root finding and cycle detection
    root_map = {}
    cycles = []

    for tid in tweet_lookup:
        curr = tid
        path = []
        path_set = set()
        while curr is not None:
            if curr in path_set:
                # Cycle detected
                cycle_start = path.index(curr)
                cycles.append((tid, path[cycle_start:] + [curr]))
                break
            if curr in root_map:
                curr = root_map[curr]
                break
            path.append(curr)
            path_set.add(curr)
            parent = parent_map.get(curr)
            # If parent is None or missing from dataset, curr is the root
            if parent is None or parent not in tweet_lookup:
                break
            curr = parent
        root = curr
        for p in path:
            root_map[p] = root

    # Group into threads
    thread_groups = defaultdict(list)
    for tid, root in root_map.items():
        thread_groups[root].append(tid)

    # Order turns within each thread (BFS traversal from root ensuring parent precedes children)
    threads = {}
    threads_with_branches = 0

    for root, tids in thread_groups.items():
        tids_set = set(tids)
        ordered_tids = []
        queue = [root]
        visited = {root}

        while queue:
            curr = queue.pop(0)
            ordered_tids.append(curr)
            kids = [c for c in children_map.get(curr, []) if c in tids_set and c not in visited]
            # Sort kids deterministically by timestamp or tweet_id
            kids.sort(key=lambda x: (tweet_lookup[x]["created_at"], x))
            for k in kids:
                visited.add(k)
                queue.append(k)

        # Handle any remaining unvisited nodes
        remaining = [t for t in tids if t not in visited]
        if remaining:
            remaining.sort(key=lambda x: (tweet_lookup[x]["created_at"], x))
            ordered_tids.extend(remaining)

        # Check branching in this thread
        has_branch = any(len([c for c in children_map.get(t, []) if c in tids_set]) > 1 for t in tids)
        if has_branch:
            threads_with_branches += 1

        # Build turn records
        turns = []
        for idx, tid in enumerate(ordered_tids):
            tw_info = dict(tweet_lookup[tid])
            tw_info["turn_index"] = idx
            turns.append(tw_info)

        threads[root] = turns

    stats = {
        "missing_parents_count": missing_parents_count,
        "missing_children_count": missing_children_count,
        "threads_with_branches": threads_with_branches,
    }

    return threads, cycles, stats


class TestConversationReconstruction(unittest.TestCase):
    """Test suite for conversation reconstruction rules and edge cases."""

    def test_simple_two_turn_conversation(self):
        """Test simple 2-tweet conversation: A -> B."""
        tweets = [
            {"tweet_id": 1, "in_response_to_tweet_id": None, "inbound": True, "text": "Help needed"},
            {"tweet_id": 2, "in_response_to_tweet_id": 1, "inbound": False, "text": "We are here to help"},
        ]
        threads, cycles, stats = reconstruct_conversation_graph(tweets)

        self.assertEqual(len(threads), 1)
        self.assertIn(1, threads)
        self.assertEqual(len(cycles), 0)
        self.assertEqual(len(threads[1]), 2)
        self.assertEqual([turn["tweet_id"] for turn in threads[1]], [1, 2])
        self.assertEqual(threads[1][0]["turn_index"], 0)
        self.assertEqual(threads[1][1]["turn_index"], 1)
        self.assertEqual(threads[1][0]["speaker"], "customer")
        self.assertEqual(threads[1][1]["speaker"], "AppleSupport")

    def test_three_turn_conversation(self):
        """Test 3-turn chain: A -> B -> C."""
        tweets = [
            {"tweet_id": 100, "in_response_to_tweet_id": None, "inbound": True, "text": "iOS issue"},
            {"tweet_id": 101, "in_response_to_tweet_id": 100, "inbound": False, "text": "What version?"},
            {"tweet_id": 102, "in_response_to_tweet_id": 101, "inbound": True, "text": "iOS 11.1"},
        ]
        threads, cycles, stats = reconstruct_conversation_graph(tweets)

        self.assertEqual(len(threads), 1)
        self.assertIn(100, threads)
        self.assertEqual([turn["tweet_id"] for turn in threads[100]], [100, 101, 102])
        self.assertEqual([turn["turn_index"] for turn in threads[100]], [0, 1, 2])

    def test_missing_parent_reference(self):
        """Test missing parent: B replies to A, but A is absent from dataset."""
        tweets = [
            {"tweet_id": 200, "in_response_to_tweet_id": 999, "inbound": False, "text": "Replying to unknown tweet"},
            {"tweet_id": 201, "in_response_to_tweet_id": 200, "inbound": True, "text": "Follow up"},
        ]
        threads, cycles, stats = reconstruct_conversation_graph(tweets)

        self.assertEqual(len(threads), 1)
        # Root is 200 because 999 is absent
        self.assertIn(200, threads)
        self.assertEqual([turn["tweet_id"] for turn in threads[200]], [200, 201])
        self.assertEqual(stats["missing_parents_count"], 1)

    def test_branching_conversation(self):
        """Test branching: A -> B and A -> C."""
        tweets = [
            {"tweet_id": 300, "in_response_to_tweet_id": None, "inbound": True, "text": "My phone is frozen"},
            {"tweet_id": 301, "in_response_to_tweet_id": 300, "inbound": False, "text": "Try force restarting", "created_at": "2017-01-01 10:00:00"},
            {"tweet_id": 302, "in_response_to_tweet_id": 300, "inbound": False, "text": "DM us here", "created_at": "2017-01-01 10:05:00"},
        ]
        threads, cycles, stats = reconstruct_conversation_graph(tweets)

        self.assertEqual(len(threads), 1)
        self.assertIn(300, threads)
        self.assertEqual(len(threads[300]), 3)
        self.assertEqual(stats["threads_with_branches"], 1)
        # Parent 300 must be first (turn 0), followed by child branches
        self.assertEqual(threads[300][0]["tweet_id"], 300)
        child_ids = {threads[300][1]["tweet_id"], threads[300][2]["tweet_id"]}
        self.assertEqual(child_ids, {301, 302})

    def test_cycle_detection(self):
        """Test cycle handling: A -> B -> A."""
        tweets = [
            {"tweet_id": 401, "in_response_to_tweet_id": 402, "inbound": True, "text": "Cycle 1"},
            {"tweet_id": 402, "in_response_to_tweet_id": 401, "inbound": False, "text": "Cycle 2"},
        ]
        threads, cycles, stats = reconstruct_conversation_graph(tweets)

        # Cycles should be detected without crashing or looping infinitely
        self.assertGreaterEqual(len(cycles), 1)
        # All tweets are still preserved in threads
        total_turns = sum(len(t) for t in threads.values())
        self.assertEqual(total_turns, 2)

    def test_standalone_tweet(self):
        """Test standalone tweet with no parent or responses."""
        tweets = [
            {"tweet_id": 500, "in_response_to_tweet_id": None, "response_tweet_id": None, "inbound": True, "text": "Standalone"}
        ]
        threads, cycles, stats = reconstruct_conversation_graph(tweets)

        self.assertEqual(len(threads), 1)
        self.assertIn(500, threads)
        self.assertEqual(len(threads[500]), 1)
        self.assertEqual(threads[500][0]["tweet_id"], 500)
        self.assertEqual(threads[500][0]["turn_index"], 0)


if __name__ == "__main__":
    unittest.main()
