"""PCB routing demonstration and reproducible two-board benchmark.

Run: python differential_pair_router_v2.py
Only the Python standard library is required. Edit SETTINGS below, not CLI flags.
See README.txt in the output directory for the scope of the demonstration.
"""

from __future__ import annotations

import csv
import heapq
import itertools
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# SETTINGS: physical dimensions are millimetres.
GRID = 0.5
TRACE_WIDTH = 0.25
CLEARANCE = 0.25             # copper-edge clearance, including unrelated pads
PAIR_EDGE_GAP = 0.75        # demonstration value, not an impedance calculation
PAIR_CENTRE_SPACING = TRACE_WIDTH + PAIR_EDGE_GAP  # 1.0 mm
PAIR_BEND_RADIUS = 1.0      # centreline; inner/outer radii 0.5 / 1.5 mm
PAIR_SKEW_TOLERANCE = 0.001
VIA_DIAMETER = 1.0
VIA_COST_MM = 6.0
BOARD_SCALES = (1, 10)      # 150x100 mm and 1500x1000 mm
BENCHMARK_REPEATS = 5
RANDOM_NET_COUNT = 25
RANDOM_SEED = 7
MAX_SEARCH_STATES = 500_000 # explicit failure, never a silently accepted partial path
MAX_SINGLE_SEARCH_STATES = 100_000 # bounded attempts for unrelated random connections
OUTPUT_DIR = Path(__file__).resolve().parent / "differential_pair_results"
PAIR_P = (22, 76)          # global zero-based pads: U1 -> J2
PAIR_N = (23, 77)
PAIR_IDS = frozenset(PAIR_P + PAIR_N)
EPS = 1e-9
ARC_STEP = math.pi / 48
ARC_ERROR = 0.001          # bounds chord approximation error for these radii


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    den = dx * dx + dy * dy
    if den < EPS:
        return distance(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / den))
    return distance(p, (a[0] + t * dx, a[1] + t * dy))


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segment_distance(a, b, c, d):
    ab_c, ab_d = cross(a, b, c), cross(a, b, d)
    cd_a, cd_b = cross(c, d, a), cross(c, d, b)
    if ab_c * ab_d < -EPS and cd_a * cd_b < -EPS:
        return 0.0
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def segment_rect_distance(a, b, box):
    x0, y0, x1, y1 = box
    if any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in (a, b)):
        return 0.0
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return min(segment_distance(a, b, corners[i], corners[(i + 1) % 4]) for i in range(4))


@dataclass(frozen=True)
class Pad:
    idx: int
    comp: str
    x: float
    y: float
    w: float
    h: float
    layer: int | None
    thru: bool = False

    @property
    def point(self):
        return self.x, self.y

    @property
    def box(self):
        return self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2


@dataclass(frozen=True)
class Primitive:
    """A physical copper centreline: a line or exact circular arc."""
    a: tuple
    b: tuple
    layer: int = 0
    center: tuple | None = None
    sweep: float = 0.0

    @property
    def length(self):
        return distance(self.a, self.b) if self.center is None else distance(self.a, self.center) * abs(self.sweep)

    def points(self):
        if self.center is None:
            return (self.a, self.b)
        r = distance(self.a, self.center)
        angle = math.atan2(self.a[1] - self.center[1], self.a[0] - self.center[0])
        count = max(2, math.ceil(abs(self.sweep) / ARC_STEP))
        return tuple((self.center[0] + r * math.cos(angle + self.sweep * i / count),
                      self.center[1] + r * math.sin(angle + self.sweep * i / count))
                     if 0 < i < count else (self.a if i == 0 else self.b)
                     for i in range(count + 1))


class SpatialIndex:
    """Sparse millimetre tiles, not a W*H grid or universal-highway dictionary."""
    def __init__(self, tile=5.0):
        self.tile = tile
        self.bins = defaultdict(list)
        self.entries = []

    def keys(self, box):
        return itertools.product(range(math.floor(box[0] / self.tile), math.floor(box[2] / self.tile) + 1),
                                 range(math.floor(box[1] / self.tile), math.floor(box[3] / self.tile) + 1))

    def insert(self, box, value):
        index = len(self.entries)
        self.entries.append(value)
        for key in self.keys(box):
            self.bins[key].append(index)

    def query(self, box):
        seen = set()
        for key in self.keys(box):
            for index in self.bins.get(key, ()):
                if index not in seen:
                    seen.add(index)
                    yield self.entries[index]


def expanded_box(a, b, margin):
    return min(a[0], b[0]) - margin, min(a[1], b[1]) - margin, max(a[0], b[0]) + margin, max(a[1], b[1]) + margin


class Board:
    def __init__(self, scale):
        self.scale = scale
        self.width, self.height = 150.0 * scale, 100.0 * scale
        self.pads, self.components = [], []
        self.pad_index = SpatialIndex()
        self.copper = [SpatialIndex(), SpatialIndex()]
        self.routes, self.vias = [], []
        self.rng = random.Random(RANDOM_SEED)
        self.place_parts()
        for pad in self.pads:
            self.pad_index.insert(pad.box, pad)

    def add_pad(self, comp, x, y, w, h, layer, thru=False):
        self.pads.append(Pad(len(self.pads), comp, round(x / GRID) * GRID,
                             round(y / GRID) * GRID, w, h, layer, thru))

    def place_parts(self):
        # Original pad creation order is preserved, including indices 22,23,76,77.
        cx, cy = 75.0 * self.scale, 50.0 * self.scale
        self.components.append(dict(name="U1", x=cx, y=cy, w=24, h=24, layer=0))
        for side in ("top", "bottom", "left", "right"):
            for i in range(12):
                if side in ("top", "bottom"):
                    x, y = cx - 11 + 2 * i, cy + (13.5 if side == "top" else -13.5)
                    w, h = 1.2, 3.0
                else:
                    x, y = cx + (-13.5 if side == "left" else 13.5), cy - 11 + 2 * i
                    w, h = 3.0, 1.2
                self.add_pad("U1", x, y, w, h, 0)
        for name, x, y, cols in (("J1", 25, 75, 10), ("J2", 125, 25, 8)):
            cx, cy = x * self.scale, y * self.scale
            self.components.append(dict(name=name, x=cx, y=cy, w=(cols - 1) * 4 + 4, h=8, layer=None))
            for row in range(2):
                for col in range(cols):
                    self.add_pad(name, cx - (cols - 1) * 2 + col * 4, cy - 2 + row * 4, 2, 2, None, True)
        for i in range(8):
            cx, cy = self.rng.randint(20, 130) * self.scale, self.rng.randint(20, 80) * self.scale
            layer = i % 2
            self.components.append(dict(name=f"R{i + 1}", x=cx, y=cy, w=6, h=3, layer=layer))
            for x in (cx - 1.5, cx + 1.5):
                self.add_pad(f"R{i + 1}", x, cy, 2, 2, layer)

    def segment_clear(self, a, b, layer, excluded=frozenset(), radius=TRACE_WIDTH / 2, arc_margin=0.0):
        # Copper-to-board-edge clearance is also enforced.
        edge = radius + CLEARANCE + arc_margin
        if any(not (edge - EPS <= p[0] <= self.width - edge + EPS and
                    edge - EPS <= p[1] <= self.height - edge + EPS) for p in (a, b)):
            return False
        required = radius + CLEARANCE + arc_margin
        for pad in self.pad_index.query(expanded_box(a, b, required)):
            if pad.idx in excluded or (pad.layer is not None and pad.layer != layer):
                continue
            actual = (point_segment_distance(pad.point, a, b) - pad.w / 2
                      if pad.thru else segment_rect_distance(a, b, pad.box))
            if actual < required - EPS:
                return False
        for c, d, copper_radius, owner in self.copper[layer].query(expanded_box(a, b, required)):
            if segment_distance(a, b, c, d) < required + copper_radius - EPS:
                return False
        return True

    def primitive_clear(self, primitive, excluded=frozenset()):
        points = primitive.points()
        margin = ARC_ERROR if primitive.center is not None else 0
        return all(self.segment_clear(a, b, primitive.layer, excluded, arc_margin=margin)
                   for a, b in zip(points, points[1:]))

    def via_clear(self, point, excluded=frozenset()):
        # Via holes cannot be dropped into even an endpoint pad in this demo.
        return all(self.segment_clear(point, point, layer, frozenset(), VIA_DIAMETER / 2)
                   for layer in (0, 1))

    def commit(self, name, primitives, endpoints, vias=()):
        for primitive in primitives:
            points = primitive.points()
            # Arc chords have a conservative occupied radius.
            radius = TRACE_WIDTH / 2 + (ARC_ERROR if primitive.center is not None else 0)
            for a, b in zip(points, points[1:]):
                self.copper[primitive.layer].insert(expanded_box(a, b, radius), (a, b, radius, name))
        for point in vias:
            self.vias.append(point)
            for layer in (0, 1):
                self.copper[layer].insert(expanded_box(point, point, VIA_DIAMETER / 2),
                                          (point, point, VIA_DIAMETER / 2, name))
        self.routes.append(dict(name=name, primitives=primitives, endpoints=endpoints, vias=list(vias)))


def length(primitives):
    return sum(p.length for p in primitives)


DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def rail_point(center, heading, side):
    dx, dy = DIRECTIONS[heading]
    half = PAIR_CENTRE_SPACING / 2
    return center[0] - side * dy * half, center[1] + side * dx * half


def pair_move(center, heading, turn):
    """One joint action, returning both physical rails and a new centre/heading."""
    dx, dy = DIRECTIONS[heading]
    if turn == 0:
        endpoint = center[0] + dx * GRID, center[1] + dy * GRID
        rails = tuple(Primitive(rail_point(center, heading, side), rail_point(endpoint, heading, side))
                      for side in (-1, 1))
        return endpoint, heading, rails, GRID
    new_heading = (heading + turn) % 4
    nx, ny = DIRECTIONS[new_heading]
    r = PAIR_BEND_RADIUS
    endpoint = center[0] + r * (dx + nx), center[1] + r * (dy + ny)
    arc_center = center[0] - turn * dy * r, center[1] + turn * dx * r
    rails = tuple(Primitive(rail_point(center, heading, side), rail_point(endpoint, new_heading, side),
                            0, arc_center, turn * math.pi / 2) for side in (-1, 1))
    return endpoint, new_heading, rails, r * math.pi / 2


def escape_pair(board):
    """Symmetric 45-degree fanouts for U1-bottom -> J2-top; P stays the west rail."""
    source = [board.pads[i] for i in (PAIR_P[0], PAIR_N[0])]
    target = [board.pads[i] for i in (PAIR_P[1], PAIR_N[1])]
    if any(p.layer not in (0, None) for p in source + target):
        raise ValueError("This pair router supports top-layer endpoints only.")
    if source[0].x >= source[1].x or target[0].x >= target[1].x:
        raise ValueError("P/N pad ordering changed: update the symmetric escape geometry.")
    half = PAIR_CENTRE_SPACING / 2
    start_x = (source[0].x + source[1].x) / 2
    goal_x = (target[0].x + target[1].x) / 2
    source_spread = (source[1].x - source[0].x) / 2 - half
    target_spread = (target[1].x - target[0].x) / 2 - half
    if min(source_spread, target_spread) < 0:
        raise ValueError("Configured rail spacing exceeds pad spacing.")
    start = start_x, source[0].y - 2.0 - source_spread
    goal = goal_x, target[0].y + 2.5 + target_spread
    fanouts = []
    for i, side in enumerate((-1, 1)):
        a, b = source[i], target[i]
        middle_a = (a.x, a.y - 2.0)
        middle_b = (b.x, b.y + 2.5)
        first = [Primitive(a.point, middle_a), Primitive(middle_a, (start_x + side * half, start[1]))]
        last = [Primitive((goal_x + side * half, goal[1]), middle_b), Primitive(middle_b, b.point)]
        own = frozenset((a.idx, b.idx))
        if not all(board.primitive_clear(p, own) for p in first + last):
            raise RuntimeError("A symmetric endpoint fanout is blocked; change placement or escape geometry.")
        fanouts.append((first, last))
    for value in start + goal:
        if abs(value / GRID - round(value / GRID)) > EPS:
            raise ValueError("Endpoint fanout does not lie on the centreline grid.")
    return start, goal, fanouts


def route_pair(board):
    """Joint weighted A*: position, heading and signed turn count.

    Both rails are checked for every action. Signed turn count must return to
    zero: inner/outer arc length differences then cancel exactly. This is a
    feasible-path search, with no shortest-path guarantee. There is no fallback
    to independent single-ended routing and no unverified length tuning.
    """
    start, goal, fanouts = escape_pair(board)
    # The coupled path must also avoid crowding the opposite signal's endpoint
    # fanouts. Checking only pads would allow a detour to double back too close
    # to those already-defined copper segments.
    fanout_index = [SpatialIndex(), SpatialIndex()]
    same_fanout_index = [SpatialIndex(), SpatialIndex()]
    for i, (first, last) in enumerate(fanouts):
        for primitive in first + last:
            fanout_index[i].insert(expanded_box(primitive.a, primitive.b, 0), (primitive.a, primitive.b))
        for j, primitive in enumerate(first + last):
            a, b = primitive.a, primitive.b
            span = distance(a, b)
            # Leave room for the intentional, continuous joint at each end.
            trim = min(2 * TRACE_WIDTH, span)
            if j == len(first) - 1:
                b = (b[0] + (a[0] - b[0]) * trim / span, b[1] + (a[1] - b[1]) * trim / span)
            elif j == len(first):
                a = (a[0] + (b[0] - a[0]) * trim / span, a[1] + (b[1] - a[1]) * trim / span)
            same_fanout_index[i].insert(expanded_box(a, b, 0), (a, b))

    def rails_clear(rails):
        for i, primitive in enumerate(rails):
            if not board.primitive_clear(primitive):
                return False
            pts = primitive.points()
            minimum = PAIR_CENTRE_SPACING + (ARC_ERROR if primitive.center is not None else 0)
            for a, b in zip(pts, pts[1:]):
                for c, d in fanout_index[1 - i].query(expanded_box(a, b, minimum)):
                    if segment_distance(a, b, c, d) < minimum - 2 * ARC_ERROR - EPS:
                        return False
                for c, d in same_fanout_index[i].query(expanded_box(a, b, TRACE_WIDTH + ARC_ERROR)):
                    if segment_distance(a, b, c, d) < TRACE_WIDTH + ARC_ERROR - EPS:
                        return False
        return True

    # Last value tracks consecutive straight moves, capped at 4 = 2 mm.
    state0 = (round(start[0] / GRID), round(start[1] / GRID), 3, 0, 0)
    goal_xy = (round(goal[0] / GRID), round(goal[1] / GRID))
    counter = itertools.count()
    heap, best, parent = [], {state0: 0.0}, {}
    collisions = {}
    expanded = 0

    def heuristic(state):
        # Deliberately weighted, greedy guidance; no optimality claim.
        return 1.10 * GRID * (abs(state[0] - goal_xy[0]) + abs(state[1] - goal_xy[1]))

    heapq.heappush(heap, (heuristic(state0), next(counter), 0.0, state0))
    while heap:
        _, _, cost, state = heapq.heappop(heap)
        if cost > best.get(state, math.inf) + EPS:
            continue
        expanded += 1
        if expanded > MAX_SEARCH_STATES:
            raise RuntimeError("Differential search state limit reached; no route accepted.")
        x, y, heading, winding,straight_run= state
        if (x, y) == goal_xy and heading == 3 and winding == 0:
            actions = []
            while state != state0:
                previous, rails = parent[state]
                actions.append(rails)
                state = previous
            actions.reverse()
            p = fanouts[0][0] + [a[0] for a in actions] + fanouts[0][1]
            n = fanouts[1][0] + [a[1] for a in actions] + fanouts[1][1]
            return p, n, dict(expanded_states=expanded, discovered_states=len(best),
                              coupled_actions=len(actions), algorithm="joint weighted A*, top layer, exact offset arcs")
        center = x * GRID, y * GRID
 
        for turn in (0, 1, -1):
            if turn != 0 and straight_run < 4:
                continue
            
            new_winding = winding + turn
            if abs(new_winding) > 2:
                continue
            endpoint, new_heading, rails, move_length = pair_move(center, heading, turn)
            new_straight_run=min(8,straight_run+1) if turn == 0 else 0
            next_state = (round(endpoint[0] / GRID),
    round(endpoint[1] / GRID),
    new_heading,
    new_winding,
    new_straight_run,)
            short_run_penalty = 0.25 * max(0, 8 - straight_run) if turn else 0.0
            TURN_PENALITY=2.0
            new_cost = cost + move_length + (TURN_PENALITY if turn else 0.0) + short_run_penalty
            if new_cost >= best.get(next_state, math.inf) - EPS:
                continue
            # Main paired corridor must avoid ALL pads, including its endpoints.
            # The separately validated fanouts provide legal pad entry/exit.
            key = x, y, heading, turn
            if key not in collisions:
                collisions[key] = rails_clear(rails)
            if not collisions[key]:
                continue
            best[next_state] = new_cost
            parent[next_state] = state, rails
            heapq.heappush(heap, (new_cost + heuristic(next_state), next(counter), new_cost, next_state))
    raise RuntimeError("No valid differential route exists with the configured same-layer actions.")


def flattened(primitives):
    return [(a, b, p.layer) for p in primitives for a, b in zip(p.points(), p.points()[1:])]


def validate_pair(board, p, n, search_stats):
    """Post-search checks independent of the A* search/collision cache."""
    for rail, ends in ((p, PAIR_P), (n, PAIR_N)):
        assert distance(rail[0].a, board.pads[ends[0]].point) < EPS, "Wrong driver endpoint"
        assert distance(rail[-1].b, board.pads[ends[1]].point) < EPS, "Wrong receiver endpoint"
        assert all(distance(a.b, b.a) < 1e-7 for a, b in zip(rail, rail[1:])), "Discontinuous route"
        assert all(primitive.layer == 0 for primitive in rail), "Pair changed layer"
        assert all(board.primitive_clear(primitive, frozenset(ends)) for primitive in rail), "Pad/copper collision"
        # Independent continuous checks for a rail crossing itself. Nearby
        # portions along a smooth curve naturally touch, so exclude immediate
        # neighbours by their distance ALONG the route, not by segment count.
        seen = SpatialIndex()
        station = 0.0
        for a, b, _ in flattened(rail):
            next_station = station + distance(a, b)
            for c, d, previous_end in seen.query(expanded_box(a, b, TRACE_WIDTH)):
                if station - previous_end <= 2 * TRACE_WIDTH + EPS:
                    continue
                if segment_distance(a, b, c, d) < TRACE_WIDTH - 2 * ARC_ERROR:
                    raise AssertionError('Rail crosses itself')
            seen.insert(expanded_box(a, b, 0), (a, b, next_station))
            station = next_station
    lp, ln = length(p), length(n)
    assert abs(lp - ln) <= PAIR_SKEW_TOLERANCE, "Pair exceeds length mismatch tolerance"
    pp, nn = flattened(p), flattened(n)
    minimum = math.inf
    # Spatial rejection keeps this independent continuous segment test sparse.
    other = SpatialIndex()
    for a, b, layer in nn:
        other.insert(expanded_box(a, b, PAIR_CENTRE_SPACING + 2 * ARC_ERROR), (a, b))
    for a, b, layer in pp:
        for c, d in other.query(expanded_box(a, b, PAIR_CENTRE_SPACING + 2 * ARC_ERROR)):
            minimum = min(minimum, segment_distance(a, b, c, d))
    # Chords on two concentric arcs can differ from ideal spacing by <=2*ARC_ERROR.
    assert minimum >= PAIR_CENTRE_SPACING - 2 * ARC_ERROR - 1e-7, "Pair spacing or crossover failure"
    assert minimum - TRACE_WIDTH >= CLEARANCE, "Pair copper short/clearance failure"
    # Verify exact spacing primitive by primitive on the coupled part.
    for a, b in zip(p[2:-2], n[2:-2]):
        assert (a.center is None) == (b.center is None)
        if a.center is None:
            assert abs(distance(a.a, b.a) - PAIR_CENTRE_SPACING) < 1e-7
            assert abs(distance(a.b, b.b) - PAIR_CENTRE_SPACING) < 1e-7
        else:
            assert distance(a.center, b.center) < 1e-7
            assert abs(abs(distance(a.a, a.center) - distance(b.a, b.center)) - PAIR_CENTRE_SPACING) < 1e-7
    return dict(p_length_mm=lp, n_length_mm=ln, skew_mm=abs(lp - ln),
                coupled_centre_spacing_mm=PAIR_CENTRE_SPACING, coupled_edge_gap_mm=PAIR_EDGE_GAP,
                sampled_min_centre_spacing_mm=minimum, pair_vias=0,
                checks="PASS: endpoints, continuity, same layer, pad/copper clearance, self-crossings, spacing, skew",
                **search_stats)


SINGLE_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))


def generate_nets(board):
    # No pad belongs to two separately generated nets; pair pads are reserved.
    available = [p.idx for p in board.pads if p.idx not in PAIR_IDS]
    rng = random.Random(700)
    nets = []
    for _ in range(3000):
        if len(nets) == RANDOM_NET_COUNT or len(available) < 2:
            break
        a, b = rng.sample(available, 2)
        pa, pb = board.pads[a], board.pads[b]
        if pa.comp == pb.comp or distance(pa.point, pb.point) < 10:
            continue
        nets.append((a, b))
        available.remove(a)
        available.remove(b)
    assert not (PAIR_IDS & {p for net in nets for p in net})
    return nets


def route_single(board, ia, ib):
    """Grid A* for the remaining random nets, with physical copper clearance."""
    pa, pb = board.pads[ia], board.pads[ib]
    excluded = frozenset((ia, ib))
    # Direction in state retains the original 45-degree bend rule.
    starts = [(round(pa.x / GRID), round(pa.y / GRID), layer, 0, 0)
              for layer in ((0, 1) if pa.thru else (pa.layer,))]
    goal = round(pb.x / GRID), round(pb.y / GRID)
    goal_layers = (0, 1) if pb.thru else (pb.layer,)
    best, parents = {s: 0.0 for s in starts}, {}
    count = itertools.count()
    heap = []
    expanded = 0
    collision_cache, via_cache = {}, {}
    
    

    def h(state):
        dx, dy = abs(state[0] - goal[0]), abs(state[1] - goal[1])
        return 1.15 * GRID * (max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy))

    for state in starts:
        heapq.heappush(heap, (h(state), next(count), 0.0, state))
    while heap:
        _, _, cost, state = heapq.heappop(heap)
        if cost > best.get(state, math.inf) + EPS:
            continue
        expanded += 1
        if ia == 56 and ib == 18 and expanded <= 3:
            print(
        "NET", ia, ib,
        "state", state,
        "pad", (pa.x, pa.y),
        "goal", goal,
        "expanded", expanded
    )
            
            

            
        
        if expanded > MAX_SINGLE_SEARCH_STATES:
            return None, dict(reason="state limit", expanded_states=expanded)
        x, y, layer, pdx, pdy = state
        point = x * GRID, y * GRID
        if (x, y) == goal and layer in goal_layers:
            path = [state]
            while state in parents:
                state = parents[state]
                path.append(state)
            path.reverse()
            primitives, vias = [], []
            for a, b in zip(path, path[1:]):
                axy, bxy = (a[0] * GRID, a[1] * GRID), (b[0] * GRID, b[1] * GRID)
                if a[2] != b[2]:
                    vias.append(axy)
                else:
                    primitives.append(Primitive(axy, bxy, a[2]))
            return (primitives, vias), dict(expanded_states=expanded)
        candidates = []
        if ia == 91 and ib == 21:
            print("START", point, "layer", layer, "pad", (pa.x, pa.y))
            for dx, dy in SINGLE_DIRS:
                end = (x + dx) * GRID, (y + dy) * GRID
                print("MOVE", (dx, dy), "to", end,
                      "clear:", board.segment_clear(point, end, layer, excluded))
            print("VIA CLEAR:", board.via_clear(point))
        
        for dx, dy in SINGLE_DIRS:
            if (pdx or pdy) and (dx, dy) != (pdx, pdy):
                if pdx * dx + pdy * dy <= 0 or pdx * pdx + pdy * pdy == dx * dx + dy * dy:
                    continue
            end = (x + dx) * GRID, (y + dy) * GRID
            key = x, y, layer, dx, dy
            if key not in collision_cache:
                collision_cache[key] = board.segment_clear(point, end, layer, excluded)
            if collision_cache[key]:
                candidates.append(((x + dx, y + dy, layer, dx, dy), GRID * math.hypot(dx, dy)))
        # Avoid testing expensive vias until ordinary moves have been considered.
        vkey = x, y
        if vkey not in via_cache:
            via_cache[vkey] = board.via_clear(point)
        if via_cache[vkey]:
            candidates.append(((x, y, 1 - layer, 0, 0), VIA_COST_MM))
        for nxt, step in candidates:
            new_cost = cost + step
            if new_cost >= best.get(nxt, math.inf) - EPS:
                continue
            best[nxt], parents[nxt] = new_cost, state
            heapq.heappush(heap, (new_cost + h(nxt), next(count), new_cost, nxt))
    return None, dict(reason="no path", expanded_states=expanded)


def validate_single(board, ia, ib, primitives, vias):
    assert primitives, "Empty single-ended route"
    pa, pb = board.pads[ia], board.pads[ib]
    assert distance(primitives[0].a, pa.point) < EPS
    assert distance(primitives[-1].b, pb.point) < EPS
    assert all(board.primitive_clear(p, frozenset((ia, ib))) for p in primitives)
    assert all(board.via_clear(p) for p in vias)


def polyline_points(primitives):
    points = []
    for p in primitives:
        pts = p.points()
        if points and distance(points[-1], pts[0]) < EPS:
            points.extend(pts[1:])
        else:
            points.extend(pts)
    return points


def export_html(board, metrics, path):
    # Standalone HTML/SVG: no Bokeh install, browser server or CDN is needed.
    payload = dict(width=board.width, height=board.height, grid=GRID,
                   components=board.components,
                   pads=[dict(idx=p.idx, comp=p.comp, x=p.x, y=p.y, w=p.w, h=p.h, layer=p.layer, thru=p.thru)
                         for p in board.pads],
                   routes=[dict(name=r['name'], endpoints=r['endpoints'],
                                segments=[dict(points=p.points(), layer=p.layer) for p in r['primitives']])
                           for r in board.routes], vias=board.vias, metrics=metrics)
    page = '''<!doctype html><meta charset="utf-8"><title>Differential PCB routing</title>
<style>body{margin:0;background:#121922;color:#e5ecf3;font:14px system-ui}header{padding:16px 22px;background:#1c2835}h2{margin:0 0 8px}button,label{margin-right:14px}button{padding:7px 12px;cursor:pointer}#note{margin-top:10px;color:#b6c8d8}svg{display:block;width:100%;height:80vh;touch-action:none;background:#0c131b}text{font-family:system-ui;paint-order:stroke;stroke:#101a24;stroke-width:.1}#stats{padding:12px 22px;white-space:pre-wrap}button:focus{outline:2px solid #5faaff}</style>
<header><h2 id="title"></h2><button id="full">Full board</button><button id="pair">Fit differential pair</button>
<label><input type="checkbox" id="top" checked>Top</label><label><input type="checkbox" id="bottom" checked>Bottom</label><label><input type="checkbox" id="other" checked>Other nets</label>
<div id="note">Wheel to zoom; drag to pan. P: red, N: cyan. Hover a pad for its global index. Copper widths are in millimetres.</div></header>
<svg id="board" xmlns="http://www.w3.org/2000/svg"></svg><div id="stats"></div>
<script>const D=__DATA__; const svg=document.getElementById('board'),NS='http://www.w3.org/2000/svg';
let view=[0,0,D.width,D.height],down=null;
function elem(type,attr,parent=svg){let e=document.createElementNS(NS,type);for(let [k,v]of Object.entries(attr))e.setAttribute(k,v);parent.appendChild(e);return e}
function apply(){svg.setAttribute('viewBox',view.join(' '))}
function draw(){svg.replaceChildren();let g=elem('g',{transform:`translate(0 ${D.height}) scale(1 -1)`});
elem('rect',{x:0,y:0,width:D.width,height:D.height,fill:'#16232a',stroke:'#8aa9aa','stroke-width':.15},g);
for(let c of D.components)elem('rect',{x:c.x-c.w/2,y:c.y-c.h/2,width:c.w,height:c.h,fill:'#a9b4bb','fill-opacity':.08,stroke:'#8a969c','stroke-width':.1},g);
for(let p of D.pads){if(p.layer===0&&!document.getElementById('top').checked||p.layer===1&&!document.getElementById('bottom').checked)continue;
let a={fill:p.thru?'#66b885':p.layer===0?'#d1a269':'#6078c7'};let e=p.thru?elem('circle',{...a,cx:p.x,cy:p.y,r:p.w/2},g):elem('rect',{...a,x:p.x-p.w/2,y:p.y-p.h/2,width:p.w,height:p.h},g);let t=elem('title',{},e);t.textContent=`${p.comp}: pads[${p.idx}] (${p.x}, ${p.y}) mm`;}
for(let r of D.routes){let ispair=r.name==='DIFF_P'||r.name==='DIFF_N';if(!ispair&&!document.getElementById('other').checked)continue;
for(let s of r.segments){if(!document.getElementById(s.layer===0?'top':'bottom').checked)continue;elem('polyline',{points:s.points.map(p=>p.join(',')).join(' '),fill:'none',stroke:r.name==='DIFF_P'?'#ff6176':r.name==='DIFF_N'?'#41dce5':s.layer===0?'#9a8a76':'#8b84ba','stroke-width':__WIDTH__,'stroke-linecap':'round','stroke-linejoin':'round',opacity:ispair?1:.65},g)}}
for(let p of D.vias)elem('circle',{cx:p[0],cy:p[1],r:__VIA__/2,fill:'#be7df3'},g);
for(let id of [22,23,76,77]){let p=D.pads[id];let t=elem('text',{x:p.x,y:-p.y-2,transform:'scale(1 -1)','font-size':1.3,fill:'#fff'},g);t.textContent=`${p.comp} [${p.idx}] ${id===22||id===76?'P':'N'}`;}apply();}
function fitPair(){let pts=D.routes.filter(r=>r.name.startsWith('DIFF')).flatMap(r=>r.segments.flatMap(s=>s.points));let xs=pts.map(p=>p[0]),ys=pts.map(p=>D.height-p[1]);let x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x,h=Math.max(...ys)-y;let m=Math.max(4,Math.max(w,h)*.05);view=[x-m,y-m,w+2*m,h+2*m];apply()}
document.getElementById('full').onclick=()=>{view=[0,0,D.width,D.height];apply()};document.getElementById('pair').onclick=fitPair;
for(let id of ['top','bottom','other'])document.getElementById(id).onchange=draw;
svg.onwheel=e=>{e.preventDefault();let r=svg.getBoundingClientRect(),f=e.deltaY>0?1.15:1/1.15;let fx=(e.clientX-r.left)/r.width,fy=(e.clientY-r.top)/r.height;view=[view[0]+view[2]*fx*(1-f),view[1]+view[3]*fy*(1-f),view[2]*f,view[3]*f];apply()};
svg.onpointerdown=e=>{down=[e.clientX,e.clientY,...view];svg.setPointerCapture(e.pointerId)};svg.onpointermove=e=>{if(!down)return;let r=svg.getBoundingClientRect();view=[down[2]-(e.clientX-down[0])*down[4]/r.width,down[3]-(e.clientY-down[1])*down[5]/r.height,down[4],down[5]];apply()};svg.onpointerup=()=>down=null;
document.getElementById('title').textContent=`${D.width} × ${D.height} mm board | grid ${D.grid} mm`;
document.getElementById('stats').textContent=`P length ${D.metrics.p_length_mm.toFixed(6)} mm | N length ${D.metrics.n_length_mm.toFixed(6)} mm | skew ${D.metrics.skew_mm.toFixed(9)} mm\nPair spacing ${D.metrics.coupled_centre_spacing_mm} mm centre-to-centre; ${D.metrics.coupled_edge_gap_mm} mm copper-edge gap. Fanouts widen at pads.\n${D.metrics.checks}\nOther nets ${D.metrics.single_nets_routed}/${D.metrics.single_nets_attempted}. ${D.metrics.single_nets_failed} failures are recorded explicitly.`;draw();fitPair();</script>'''
    path.write_text(page.replace('__DATA__', json.dumps(payload)).replace('__WIDTH__', str(TRACE_WIDTH)).replace('__VIA__', str(VIA_DIAMETER)), encoding='utf-8')


def run_scenario(scale):
    setup_start = time.perf_counter()
    board = Board(scale)
    setup_s = time.perf_counter() - setup_start
    # Warm-up is excluded; the board is not mutated by route_pair/validation.
    route_pair(board)
    times, validation_times, verified_times = [], [], []
    for _ in range(BENCHMARK_REPEATS):
        t0 = time.perf_counter()
        p, n, stats = route_pair(board)
        search_end = time.perf_counter()
        metrics = validate_pair(board, p, n, stats)
        checked_end = time.perf_counter()
        times.append(search_end - t0)
        validation_times.append(checked_end - search_end)
        verified_times.append(checked_end - t0)
    board.commit('DIFF_P', p, PAIR_P)
    board.commit('DIFF_N', n, PAIR_N)
    # Singles see both already-committed rails as real copper keepouts.
    nets = generate_nets(board)
    failed, singles_stats = [], []
    t0 = time.perf_counter()
    for i, (a, b) in enumerate(nets):
        result, stat = route_single(board, a, b)
        singles_stats.append(stat)
        if result is None:
            failed.append(dict(pads=[a, b], **stat))
            print(f'    NET_{i:02d} {a}->{b}: FAILED ({stat["reason"]}, {stat["expanded_states"]} states)', flush=True)
            continue
        primitives, vias = result
        validate_single(board, a, b, primitives, vias)
        board.commit(f'NET_{i:02d}', primitives, (a, b), vias)
        print(f'    NET_{i:02d} {a}->{b}: routed ({stat["expanded_states"]} states)', flush=True)
    singles_s = time.perf_counter() - t0
    metrics.update(board_scale=scale, board_width_mm=board.width, board_height_mm=board.height,
                   grid_mm=GRID, grid_columns=round(board.width / GRID), grid_rows=round(board.height / GRID),
                   grid_cells_per_layer=round(board.width / GRID) * round(board.height / GRID),
                   pair_route_seconds=times, pair_median_seconds=statistics.median(times),
                   pair_min_seconds=min(times), pair_max_seconds=max(times), setup_seconds=setup_s,
                   pair_validation_seconds=statistics.median(validation_times),
                   pair_validation_samples_seconds=validation_times,
                   pair_verified_samples_seconds=verified_times,
                   pair_verified_median_seconds=statistics.median(verified_times),
                   single_routing_and_validation_seconds=singles_s,
                   single_nets_attempted=len(nets), single_nets_routed=len(nets) - len(failed),
                   single_nets_failed=len(failed), failures=failed,
                   single_expanded_states=sum(s['expanded_states'] for s in singles_stats),
                   pairs=dict(P=list(PAIR_P), N=list(PAIR_N)),
                   endpoint_coordinates={str(i): board.pads[i].point for i in sorted(PAIR_IDS)})
    stem = 'original_board' if scale == 1 else 'large_board'
    export_html(board, metrics, OUTPUT_DIR / f'{stem}.html')
    route_data = [dict(name=r['name'], endpoints=r['endpoints'], vias=r['vias'],
                       primitives=[dict(a=x.a, b=x.b, layer=x.layer, center=x.center, sweep=x.sweep,
                                        length_mm=x.length) for x in r['primitives']]) for r in board.routes]
    (OUTPUT_DIR / f'{stem}_routes.json').write_text(json.dumps(route_data, indent=2), encoding='utf-8')
    return metrics


def configuration_checks():
    assert GRID > 0 and TRACE_WIDTH > 0 and CLEARANCE >= 0
    assert PAIR_BEND_RADIUS > PAIR_CENTRE_SPACING / 2 + TRACE_WIDTH / 2
    assert abs(PAIR_BEND_RADIUS / GRID - round(PAIR_BEND_RADIUS / GRID)) < EPS
    assert abs((PAIR_CENTRE_SPACING / 2) / GRID - round((PAIR_CENTRE_SPACING / 2) / GRID)) < EPS
    assert (PAIR_BEND_RADIUS + PAIR_CENTRE_SPACING / 2) * (1 - math.cos(ARC_STEP / 2)) <= ARC_ERROR
    # Small independent geometry checks catch crossings as well as near misses.
    assert segment_distance((0, 0), (2, 2), (0, 2), (2, 0)) == 0
    assert abs(segment_distance((0, 0), (2, 0), (0, 1), (2, 1)) - 1) < EPS
    assert segment_rect_distance((-1, .5), (2, .5), (0, 0, 1, 1)) == 0
    # All four turn directions preserve exact rail spacing and continuity.
    for heading in range(4):
        for turn in (-1, 1):
            endpoint, new_heading, rails, _ = pair_move((10, 10), heading, turn)
            assert abs(distance(rails[0].a, rails[1].a) - PAIR_CENTRE_SPACING) < EPS
            assert abs(distance(rails[0].b, rails[1].b) - PAIR_CENTRE_SPACING) < EPS
            for primitive, side in zip(rails, (-1, 1)):
                assert distance(primitive.b, rail_point(endpoint, new_heading, side)) < EPS
                assert abs(distance(primitive.a, primitive.center) - distance(primitive.b, primitive.center)) < EPS


def regression_checks():
    """Actual routing regressions, outside the benchmark timing window."""
    checks = {'geometry_and_all_turn_directions': 'PASS'}
    board = Board(1)
    obstacle = Pad(1000, 'TEST_KEEPOUT', 100, 32, 4, 4, 0)
    board.pad_index.insert(obstacle.box, obstacle)
    p, n, stats = route_pair(board)
    validate_pair(board, p, n, stats)
    checks['obstacle_detour'] = dict(result='PASS', length_mm=length(p), expanded_states=stats['expanded_states'])
    board = Board(1)
    start, goal, _ = escape_pair(board)
    obstacle = Pad(1001, 'BLOCKED_START', *start, 5, 5, 0)
    board.pad_index.insert(obstacle.box, obstacle)
    try:
        route_pair(board)
    except RuntimeError:
        checks['blocked_escape_rejected'] = 'PASS'
    else:
        raise AssertionError('Blocked pair escape was accepted')
    board = Board(1)
    p, n, stats = route_pair(board)
    bad = list(n)
    mid = len(bad) // 2
    old = bad[mid]
    bad[mid] = Primitive((old.a[0] - 1, old.a[1]), (old.b[0] - 1, old.b[1]), old.layer)
    try:
        validate_pair(board, p, bad, stats)
    except AssertionError:
        checks['broken_pair_rejected'] = 'PASS'
    else:
        raise AssertionError('Broken pair was accepted')
    for scale in BOARD_SCALES:
        nets = generate_nets(Board(scale))
        used = [i for net in nets for i in net]
        assert not (set(used) & PAIR_IDS)
        assert len(set(used)) == len(used)
    checks['reserved_pair_pads_and_unique_single_net_endpoints'] = 'PASS'
    return checks


def main():
    configuration_checks()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'verification_results.json').write_text(json.dumps(regression_checks(), indent=2), encoding='utf-8')
    results = []
    for scale in BOARD_SCALES:
        print(f'Routing {150 * scale} x {100 * scale} mm, GRID={GRID} mm...', flush=True)
        metrics = run_scenario(scale)
        results.append(metrics)
        print(f'  Pair: {metrics["p_length_mm"]:.6f}/{metrics["n_length_mm"]:.6f} mm, '
              f'search median {metrics["pair_median_seconds"]:.6f} s, '
              f'with checks {metrics["pair_verified_median_seconds"]:.6f} s, '
              f'{metrics["expanded_states"]} states; singles '
              f'{metrics["single_nets_routed"]}/{metrics["single_nets_attempted"]}', flush=True)
    report = dict(python=sys.version, platform=platform.platform(),
                  benchmark_repeats=BENCHMARK_REPEATS, warmup_runs=1,
                  timing_scope='pair_median_seconds: search, endpoint escapes and search-time collision checks. '
                               'pair_verified_median_seconds: search PLUS independent validation, five full trials. '
                               'Both exclude board setup, singles, rendering and file writes.',
                  scale_definition='Board width, height and component centres scale; footprints, pitch, grid and trace dimensions stay fixed.',
                  scenarios=results)
    if len(results) == 2:
        report['large_to_original_pair_time_ratio'] = results[1]['pair_median_seconds'] / results[0]['pair_median_seconds']
        report['large_to_original_verified_pair_time_ratio'] = results[1]['pair_verified_median_seconds'] / results[0]['pair_verified_median_seconds']
        report['large_to_original_pair_length_ratio'] = results[1]['p_length_mm'] / results[0]['p_length_mm']
    (OUTPUT_DIR / 'benchmark_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    columns = ['board_scale', 'board_width_mm', 'board_height_mm', 'grid_mm', 'grid_cells_per_layer',
               'pair_median_seconds', 'pair_verified_median_seconds', 'pair_validation_seconds',
               'pair_min_seconds', 'pair_max_seconds', 'expanded_states',
               'p_length_mm', 'n_length_mm', 'skew_mm', 'single_nets_routed', 'single_nets_failed',
               'single_routing_and_validation_seconds']
    with (OUTPUT_DIR / 'benchmark_results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f'Results: {OUTPUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
