"""Nexus Swarm — swarm behaviors, emergence detection, consensus.

Pheromone-based coordination, Reynolds flocking, consensus protocols,
emergence detection, and swarm formation management.
"""
import math, random, time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum


class ConsensusType(Enum):
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    BYZANTINE_PAXOS = "byzantine_paxos"
    CRDT_MERGE = "crdt_merge"


@dataclass
class AgentPos:
    agent_id: str
    x: float
    y: float
    heading: float = 0
    speed: float = 1.0


class PheromoneField:
    """Spatial pheromone field for stigmergic coordination."""

    def __init__(self, width: int = 100, height: int = 100,
                 cell_size: float = 1.0, decay_rate: float = 0.95):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.decay_rate = decay_rate
        self.field: List[List[float]] = [[0.0]*width for _ in range(height)]

    def deposit(self, x: float, y: float, amount: float, radius: float = 2.0) -> None:
        """Deposit pheromone at position with spatial spread."""
        cx, cy = int(x / self.cell_size), int(y / self.cell_size)
        r_cells = int(radius / self.cell_size) + 1
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    dist = math.sqrt(dx*dx + dy*dy) * self.cell_size
                    if dist <= radius:
                        falloff = 1.0 - dist / (radius + 0.001)
                        self.field[ny][nx] += amount * falloff

    def sample(self, x: float, y: float) -> float:
        cx = max(0, min(int(x / self.cell_size), self.width - 1))
        cy = max(0, min(int(y / self.cell_size), self.height - 1))
        return self.field[cy][cx]

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """Compute pheromone gradient at position."""
        s = self.cell_size
        dx = (self.sample(x + s, y) - self.sample(x - s, y)) / (2 * s)
        dy = (self.sample(x, y + s) - self.sample(x, y - s)) / (2 * s)
        return (dx, dy)

    def decay(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                self.field[y][x] *= self.decay_rate
                if self.field[y][x] < 0.001:
                    self.field[y][x] = 0


class ReynoldsFlock:
    """Boids-style flocking: separation, alignment, cohesion."""

    def __init__(self, separation_dist: float = 3.0,
                 alignment_dist: float = 10.0,
                 cohesion_dist: float = 15.0):
        self.sep_dist = separation_dist
        self.align_dist = alignment_dist
        self.coh_dist = cohesion_dist
        self.w_sep = 1.5
        self.w_align = 1.0
        self.w_coh = 1.0

    def compute(self, agent: AgentPos, neighbors: List[AgentPos]) -> Tuple[float, float]:
        sep_x, sep_y = 0, 0
        align_x, align_y = 0, 0
        coh_x, coh_y = 0, 0
        n_sep, n_align, n_coh = 0, 0, 0

        for other in neighbors:
            if other.agent_id == agent.agent_id:
                continue
            dx = other.x - agent.x
            dy = other.y - agent.y
            dist = math.sqrt(dx*dx + dy*dy)

            if 0 < dist < self.sep_dist:
                sep_x -= dx / dist
                sep_y -= dy / dist
                n_sep += 1
            if 0 < dist < self.align_dist:
                align_x += math.cos(math.radians(other.heading))
                align_y += math.sin(math.radians(other.heading))
                n_align += 1
            if 0 < dist < self.coh_dist:
                coh_x += other.x
                coh_y += other.y
                n_coh += 1

        if n_sep > 0:
            sep_x /= n_sep
            sep_y /= n_sep
        if n_align > 0:
            align_x /= n_align
            align_y /= n_align
        if n_coh > 0:
            coh_x = coh_x / n_coh - agent.x
            coh_y = coh_y / n_coh - agent.y

        fx = self.w_sep * sep_x + self.w_align * align_x + self.w_coh * coh_x
        fy = self.w_sep * sep_y + self.w_align * align_y + self.w_coh * coh_y

        return (fx, fy)


class ConsensusEngine:
    """Multi-agent consensus protocols."""

    def majority_vote(self, votes: Dict[str, str]) -> Tuple[str, float]:
        counts: Dict[str, int] = {}
        for agent_id, vote in votes.items():
            counts[vote] = counts.get(vote, 0) + 1
        if not counts:
            return ("", 0)
        winner = max(counts, key=counts.get)
        confidence = counts[winner] / len(votes)
        return (winner, confidence)

    def weighted_consensus(self, votes: Dict[str, str],
                          weights: Dict[str, float]) -> Tuple[str, float]:
        scores: Dict[str, float] = {}
        total_w = 0
        for aid, vote in votes.items():
            w = weights.get(aid, 1.0)
            scores[vote] = scores.get(vote, 0) + w
            total_w += w
        if not scores or total_w == 0:
            return ("", 0)
        winner = max(scores, key=scores.get)
        return (winner, scores[winner] / total_w)

    def byzantine_tolerance(self, votes: Dict[str, str],
                           max_faulty: int = 0) -> Tuple[Optional[str], bool]:
        """Detect Byzantine faults — at most f faulty out of 3f+1."""
        n = len(votes)
        f_max = (n - 1) // 3
        if max_faulty > 0:
            f_max = min(f_max, max_faulty)

        counts: Dict[str, int] = {}
        for vote in votes.values():
            counts[vote] = counts.get(vote, 0) + 1

        if not counts:
            return (None, False)

        # Correct nodes must agree; faulty can be anything
        non_faulty = n - f_max
        for val, count in counts.items():
            if count >= non_faulty:
                return (val, True)
        return (None, False)


class EmergenceDetector:
    """Detect emergent behaviors in swarm."""

    def __init__(self):
        self.prev_positions: Dict[str, AgentPos] = {}

    def detect_formation(self, agents: List[AgentPos]) -> Optional[str]:
        if len(agents) < 3:
            return None
        xs = [a.x for a in agents]
        ys = [a.y for a in agents]
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)

        if spread_x < 5 and spread_y < 5:
            return "cluster"
        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(agents)
        angles = [math.atan2(a.y - avg_y, a.x - avg_x) for a in agents]
        angle_spread = max(angles) - min(angles)
        if angle_spread > 5.5:
            return "circle"

        # Check alignment
        headings = [a.heading for a in agents]
        if max(headings) - min(headings) < 30:
            return "stream"

        return None

    def detect_speed_change(self, agents: List[AgentPos]) -> Dict[str, float]:
        changes = {}
        for a in agents:
            prev = self.prev_positions.get(a.agent_id)
            if prev:
                dx = a.x - prev.x
                dy = a.y - prev.y
                actual_speed = math.sqrt(dx*dx + dy*dy)
                changes[a.agent_id] = actual_speed - prev.speed
            self.prev_positions[a.agent_id] = a
        return changes


def demo():
    print("=== Swarm Behaviors ===\n")
    random.seed(42)

    # Pheromone field
    print("--- Pheromone Coordination ---")
    field = PheromoneField(50, 50, 1.0, 0.98)
    # Food source at (25, 25)
    field.deposit(25, 25, 10.0, 3.0)
    # Agent deposits trail
    for i in range(10):
        field.deposit(10 + i, 15, 2.0, 1.0)
    field.decay()

    grad = field.gradient(15, 15)
    print(f"  Gradient at (15,15): ({grad[0]:.3f}, {grad[1]:.3f})")
    print(f"  Pheromone at source (25,25): {field.sample(25, 25):.2f}")
    print(f"  Pheromone at trail (15,15): {field.sample(15, 15):.2f}")

    # Reynolds flocking
    print("\n--- Reynolds Flocking ---")
    flock = ReynoldsFlock()
    agents = [AgentPos(f"bot_{i}", random.uniform(0, 30), random.uniform(0, 30),
                       random.uniform(0, 360)) for i in range(8)]
    center = agents[0]
    neighbors = agents[1:]
    fx, fy = flock.compute(center, neighbors)
    heading = math.degrees(math.atan2(fy, fx)) % 360
    print(f"  Center agent heading influence: {heading:.1f}deg")
    print(f"  Flock force: ({fx:.2f}, {fy:.2f})")

    # Consensus
    print("\n--- Consensus ---")
    ce = ConsensusEngine()
    votes = {"a1": "north", "a2": "north", "a3": "east", "a4": "north", "a5": "east"}
    winner, conf = ce.majority_vote(votes)
    print(f"  Majority: {winner} ({conf:.0%})")

    weights = {"a1": 0.9, "a2": 0.8, "a3": 0.3, "a4": 0.7, "a5": 0.4}
    winner, conf = ce.weighted_consensus(votes, weights)
    print(f"  Weighted: {winner} ({conf:.0%})")

    # Byzantine tolerance (5 agents, 1 faulty)
    result, ok = ce.byzantine_tolerance(votes, max_faulty=1)
    print(f"  Byzantine (f=1): {result} (consensus={'yes' if ok else 'no'})")

    # Emergence
    print("\n--- Emergence Detection ---")
    ed = EmergenceDetector()
    cluster = [AgentPos(f"c{i}", 10 + random.gauss(0, 1), 10 + random.gauss(0, 1))
              for i in range(5)]
    form = ed.detect_formation(cluster)
    print(f"  Cluster agents: formation={form}")

    stream = [AgentPos(f"s{i}", i * 3, 0, heading=90) for i in range(5)]
    form = ed.detect_formation(stream)
    print(f"  Stream agents: formation={form}")


if __name__ == "__main__":
    demo()
