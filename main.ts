// sokoban-solver-01 entry: a push-optimal Sokoban solver.
//
// The generated levels are a one-square-wide hallway with small rooms
// opening off it. A box pushed out of its room can never come back, so
// each room is an independent push puzzle and the level optimum is the
// sum of the room optima. This program therefore:
//   1. splits the level into the rooms hanging off the hallway row,
//   2. solves every room with an A* search whose moves are pushes,
//   3. joins the room solutions with plain walking moves.
// If the split ever fails, a whole-level search is tried instead.
//
// Search states hold the sorted box squares and the player square; the
// transposition key replaces the player square by the lowest square of
// the region the player can walk to, which collapses all walking inside
// a state. The heuristic is a minimum-cost matching between boxes and
// goals over exact one-box push distances (computed by pulling boxes
// backwards from each goal on the empty board), so it never
// overestimates and the first solution an A* run returns is
// push-optimal for its room. A push that strands a box on a square no
// goal can be pulled back to is pruned before it is queued, and so is
// a push that freezes a 2x2 block of walls and boxes around a box
// that is not on a goal.
//
// Everything is budgeted in expanded nodes, never in seconds, and all
// iteration orders are fixed, so two runs of this program on the same
// input produce byte-identical output.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Direction tables: 0 up, 1 down, 2 left, 3 right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const OPP = [1, 0, 3, 2];
const DCH = ['u', 'd', 'l', 'r'];

// ---------------------------------------------------------------------
// Level parsing, mirroring the checker's reading of level.txt.
// ---------------------------------------------------------------------
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const n = w * h;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const boxes = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    const row = rows[y];
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  // Neighbour table; -1 means off the board (treated as a wall).
  const nb = new Int32Array(n * 4).fill(-1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d];
        const ny = y + DY[d];
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) nb[i * 4 + d] = ny * w + nx;
      }
    }
  }
  return { w, h, n, wall, goal, boxes, player, nb };
}

// ---------------------------------------------------------------------
// A binary heap of node ids ordered by (f, then deeper g, then older).
// ---------------------------------------------------------------------
class Heap {
  constructor(f, g) {
    this.f = f;
    this.g = g;
    this.a = [];
  }
  get size() {
    return this.a.length;
  }
  less(i, j) {
    const a = this.a[i];
    const b = this.a[j];
    if (this.f[a] !== this.f[b]) return this.f[a] < this.f[b];
    if (this.g[a] !== this.g[b]) return this.g[a] > this.g[b];
    return a < b;
  }
  push(v) {
    const a = this.a;
    a.push(v);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(i, p)) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && this.less(l, m)) m = l;
        if (r < a.length && this.less(r, m)) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]];
        i = m;
      }
    }
    return top;
  }
}

// ---------------------------------------------------------------------
// A push-search world: the squares boxes may occupy, the squares the
// player may walk on, and the goals to cover. Coordinates are always
// the level's own square indices.
// ---------------------------------------------------------------------
class PushWorld {
  constructor(level, boxMask, walkMask, goals) {
    this.lv = level;
    this.boxMask = boxMask;
    this.walkMask = walkMask;
    this.goals = goals;
    const n = level.n;
    // dist[g][cell]: pushes to bring one box from cell to goals[g] on
    // an otherwise empty board, -1 when impossible. Computed by
    // walking the box backwards: a box now on c could have been
    // pushed here along d from p = c - d with the player standing on
    // the square behind p.
    this.dist = goals.map((gc) => {
      const dist = new Int32Array(n).fill(-1);
      dist[gc] = 0;
      const queue = [gc];
      for (let head = 0; head < queue.length; head++) {
        const c = queue[head];
        for (let d = 0; d < 4; d++) {
          const p = level.nb[c * 4 + OPP[d]];
          if (p < 0 || dist[p] >= 0) continue;
          if (level.wall[p] || !boxMask[p]) continue;
          const s = level.nb[p * 4 + OPP[d]];
          if (s < 0 || level.wall[s] || !walkMask[s]) continue;
          dist[p] = dist[c] + 1;
          queue.push(p);
        }
      }
      return dist;
    });
    // A non-goal square no goal can be pulled back to is dead: a box
    // pushed there can never end on a goal, so such pushes are pruned.
    this.dead = new Uint8Array(n);
    for (let c = 0; c < n; c++) {
      if (!boxMask[c] || level.goal[c]) continue;
      let live = false;
      for (const dist of this.dist) {
        if (dist[c] >= 0) {
          live = true;
          break;
        }
      }
      if (!live) this.dead[c] = 1;
    }
    // Scratch buffers reused by every search in this world.
    this.stamp = new Int32Array(n);
    this.stampId = 0;
    this.floodId = 0;
    this.parent = new Int32Array(n);
    this.queue = new Int32Array(n);
  }

  // Minimum-cost matching of boxes to goals over this.dist (DP over
  // goal subsets; exact for the box counts these worlds have).
  // Returns Infinity when some box cannot reach any goal left for it.
  heuristic(boxes) {
    const k = boxes.length;
    if (k === 0) return 0;
    if (k !== this.goals.length || k > 12) {
      // Fallback (not expected in this competition): the largest
      // single-box distance is still a lower bound on the pushes left.
      let worst = 0;
      for (const b of boxes) {
        let best = Infinity;
        for (const dist of this.dist) {
          if (dist[b] >= 0 && dist[b] < best) best = dist[b];
        }
        if (best === Infinity) return Infinity;
        if (best > worst) worst = best;
      }
      return worst;
    }
    const size = 1 << k;
    let dp = new Float64Array(size).fill(Infinity);
    dp[0] = 0;
    for (let i = 0; i < k; i++) {
      const next = new Float64Array(size).fill(Infinity);
      const b = boxes[i];
      for (let mask = 0; mask < size; mask++) {
        const cur = dp[mask];
        if (cur === Infinity) continue;
        for (let g = 0; g < k; g++) {
          if (mask & (1 << g)) continue;
          const c = this.dist[g][b];
          if (c < 0) continue;
          const nm = mask | (1 << g);
          if (cur + c < next[nm]) next[nm] = cur + c;
        }
      }
      dp = next;
    }
    return dp[size - 1];
  }

  // Flood the squares the player can walk to from `start` without
  // moving a box; boxSet marks the occupied squares. Fills this.parent
  // with the direction used to first reach each square and returns the
  // lowest reachable square index (the region's canonical name).
  flood(start, boxSet) {
    const lv = this.lv;
    const id = ++this.stampId;
    const q = this.queue;
    let qh = 0;
    let qt = 0;
    q[qt++] = start;
    this.stamp[start] = id;
    this.parent[start] = -1;
    let lowest = start;
    while (qh < qt) {
      const c = q[qh++];
      for (let d = 0; d < 4; d++) {
        const nx = lv.nb[c * 4 + d];
        if (nx < 0 || this.stamp[nx] === id) continue;
        if (lv.wall[nx] || !this.walkMask[nx] || boxSet.has(nx)) continue;
        this.stamp[nx] = id;
        this.parent[nx] = d;
        if (nx < lowest) lowest = nx;
        q[qt++] = nx;
      }
    }
    this.floodId = id;
    return lowest;
  }

  reachable(cell) {
    return this.stamp[cell] === this.floodId;
  }

  // True when the box just pushed onto `dest` completes a 2x2 block
  // of walls and boxes that holds a box off its goal. No box of such a
  // block can ever move again (the player cannot stand inside the
  // block, and every push of a block box is blocked by the block
  // itself), so any state containing one is a deadlock.
  freezesBlock(dest, boxSet) {
    const lv = this.lv;
    const x = dest % lv.w;
    const y = Math.floor(dest / lv.w);
    for (const [ox, oy] of [[-1, -1], [-1, 0], [0, -1], [0, 0]]) {
      let allBlocked = true;
      let looseBox = false;
      for (let dy = 0; dy <= 1; dy++) {
        for (let dx = 0; dx <= 1; dx++) {
          const cx = x + ox + dx;
          const cy = y + oy + dy;
          if (cx < 0 || cx >= lv.w || cy < 0 || cy >= lv.h) continue;
          const c = cy * lv.w + cx;
          if (boxSet.has(c)) {
            if (!lv.goal[c]) looseBox = true;
          } else if (!lv.wall[c]) {
            allBlocked = false;
          }
        }
      }
      if (allBlocked && looseBox) return true;
    }
    return false;
  }

  // The walking moves from the current flood's start square to
  // `target`, read off the parent pointers of the current flood.
  walkTo(target) {
    const chars = [];
    let c = target;
    while (this.parent[c] >= 0) {
      const d = this.parent[c];
      chars.push(DCH[d]);
      c = this.lv.nb[c * 4 + OPP[d]];
    }
    chars.reverse();
    return chars.join('');
  }

  // A* over pushes. startBoxes is sorted; startPlayer is the player's
  // square. accept(boxes, player), when given, must also hold for a
  // goal state to count (used to demand an exit route out of a room).
  // Returns { moves, pushes, endPlayer }, or null when the expansion
  // budget runs out before any solution is found.
  solve(startBoxes, startPlayer, budget, accept) {
    const lv = this.lv;
    for (const b of startBoxes) {
      if (this.dead[b]) return null;
    }
    const h0 = this.heuristic(startBoxes);
    if (h0 === Infinity) return null;

    const nodeBoxes = [startBoxes];
    const nodePlayer = [startPlayer];
    const nodeParent = [-1];
    const nodeSeg = [''];
    const nodeG = [0];
    const nodeF = [h0];
    const best = new Map();
    const heap = new Heap(nodeF, nodeG);

    const startRep = this.flood(startPlayer, new Set(startBoxes));
    best.set(startBoxes.join(',') + '|' + startRep, 0);
    heap.push(0);
    let expansions = 0;

    while (heap.size > 0) {
      const id = heap.pop();
      const boxes = nodeBoxes[id];
      const g = nodeG[id];
      const boxSet = new Set(boxes);
      const rep = this.flood(nodePlayer[id], boxSet);
      const key = boxes.join(',') + '|' + rep;
      if (best.get(key) < g) continue; // a cheaper route here is known

      let solved = true;
      for (const b of boxes) {
        if (!lv.goal[b]) {
          solved = false;
          break;
        }
      }
      if (solved && (!accept || accept(boxes, nodePlayer[id]))) {
        const segs = [];
        for (let cur = id; nodeParent[cur] >= 0; cur = nodeParent[cur]) {
          segs.push(nodeSeg[cur]);
        }
        segs.reverse();
        return { moves: segs.join(''), pushes: g, endPlayer: nodePlayer[id] };
      }
      if (++expansions > budget) return null;

      // First pass: collect every push this state allows, reading all
      // reachability and walking segments from this node's own flood.
      // Generating a child floods again for its key and overwrites
      // that data, so nothing from the flood may be read afterwards.
      const options = [];
      for (let bi = 0; bi < boxes.length; bi++) {
        const b = boxes[bi];
        for (let d = 0; d < 4; d++) {
          const stand = lv.nb[b * 4 + OPP[d]];
          if (stand < 0 || !this.reachable(stand)) continue;
          const dest = lv.nb[b * 4 + d];
          if (dest < 0 || lv.wall[dest] || !this.boxMask[dest]) continue;
          if (boxSet.has(dest) || this.dead[dest]) continue;
          options.push({ bi, b, d, dest, seg: this.walkTo(stand) + DCH[d] });
        }
      }
      // Second pass: build the children.
      for (const { bi, b, d, dest, seg } of options) {
        const moved = boxes.slice();
        moved[bi] = dest;
        moved.sort((x, y) => x - y);
        const movedSet = new Set(moved);
        if (this.freezesBlock(dest, movedSet)) continue;
        const g2 = g + 1;
        const h2 = this.heuristic(moved);
        if (h2 === Infinity) continue;
        const rep2 = this.flood(b, movedSet);
        const key2 = moved.join(',') + '|' + rep2;
        if (best.has(key2) && best.get(key2) <= g2) continue;
        best.set(key2, g2);
        nodeBoxes.push(moved);
        nodePlayer.push(b);
        nodeParent.push(id);
        nodeSeg.push(seg);
        nodeG.push(g2);
        nodeF.push(g2 + h2);
        heap.push(nodeBoxes.length - 1);
      }
    }
    return null;
  }
}

// ---------------------------------------------------------------------
// Walking on the whole level (no pushes): a shortest move string from
// `from` to any square of `targets`, with the `blocked` squares (the
// boxes) treated as walls; null when there is no route.
// ---------------------------------------------------------------------
function walkPath(level, from, targets, blocked) {
  const n = level.n;
  const parent = new Int32Array(n).fill(-2);
  const queue = new Int32Array(n);
  let qh = 0;
  let qt = 0;
  queue[qt++] = from;
  parent[from] = -1;
  const targetSet = new Set(targets);
  let hit = targetSet.has(from) ? from : -1;
  while (qh < qt && hit < 0) {
    const c = queue[qh++];
    for (let d = 0; d < 4; d++) {
      const nx = level.nb[c * 4 + d];
      if (nx < 0 || parent[nx] !== -2) continue;
      if (level.wall[nx] || blocked.has(nx)) continue;
      parent[nx] = d;
      if (targetSet.has(nx)) {
        hit = nx;
        break;
      }
      queue[qt++] = nx;
    }
  }
  if (hit < 0) return null;
  const chars = [];
  let c = hit;
  while (parent[c] >= 0) {
    const d = parent[c];
    chars.push(DCH[d]);
    c = level.nb[c * 4 + OPP[d]];
  }
  chars.reverse();
  return { moves: chars.join(''), end: hit };
}

// ---------------------------------------------------------------------
// Splitting a level into rooms. The hallway is the player's row; the
// remaining floor squares fall into connected regions, one per room.
// A room's entry squares are the hallway squares next to its floor.
// ---------------------------------------------------------------------
function splitRooms(level) {
  const { w, n, wall } = level;
  const hallRow = Math.floor(level.player / w);
  const region = new Int32Array(n).fill(-1);
  const regions = [];
  const stack = [];
  for (let c = 0; c < n; c++) {
    if (wall[c] || Math.floor(c / w) === hallRow || region[c] >= 0) continue;
    const id = regions.length;
    const cells = [];
    stack.push(c);
    region[c] = id;
    while (stack.length > 0) {
      const cur = stack.pop();
      cells.push(cur);
      for (let d = 0; d < 4; d++) {
        const nx = level.nb[cur * 4 + d];
        if (nx < 0 || wall[nx] || region[nx] >= 0) continue;
        if (Math.floor(nx / w) === hallRow) continue;
        region[nx] = id;
        stack.push(nx);
      }
    }
    regions.push(cells);
  }
  const rooms = [];
  for (const cells of regions) {
    const cellSet = new Set(cells);
    const boxes = level.boxes.filter((b) => cellSet.has(b));
    if (boxes.length === 0) continue;
    const goals = [];
    const entries = new Set();
    for (const c of cells) {
      if (level.goal[c]) goals.push(c);
      for (let d = 0; d < 4; d++) {
        const nx = level.nb[c * 4 + d];
        if (nx >= 0 && Math.floor(nx / w) === hallRow && !wall[nx]) entries.add(nx);
      }
    }
    goals.sort((a, b) => a - b);
    rooms.push({ cells: cellSet, boxes, goals, entries: [...entries].sort((a, b) => a - b) });
  }
  return rooms;
}

// Solve one level; returns the full move string ('' when no plan was
// found: an empty answer is valid and simply scores nothing).
function solveLevel(level) {
  const rooms = splitRooms(level);
  const covered = rooms.reduce((sum, r) => sum + r.boxes.length, 0);
  if (rooms.length > 0 && covered === level.boxes.length) {
    const plan = solveByRooms(level, rooms);
    if (plan !== null) return plan;
  }
  // Fallback: one search over the whole level.
  const all = new Uint8Array(level.n);
  const goals = [];
  for (let c = 0; c < level.n; c++) {
    if (!level.wall[c]) {
      all[c] = 1;
      if (level.goal[c]) goals.push(c);
    }
  }
  const world = new PushWorld(level, all, all, goals);
  const res = world.solve(level.boxes, level.player, 1500000, null);
  return res ? res.moves : '';
}

// Solve a level room by room and stitch the pieces together with
// walking moves. Returns null when any room defeats its budget.
function solveByRooms(level, rooms) {
  const order = rooms.slice().sort((a, b) => a.entries[0] - b.entries[0]);
  const globalBoxes = new Set(level.boxes);
  let player = level.player;
  let moves = '';

  for (let i = 0; i < order.length; i++) {
    const room = order[i];
    const isLast = i === order.length - 1;
    const roomBoxSet = new Set(room.boxes);

    // The room's world: boxes stay on room squares; the player walks
    // on room squares plus the room's hallway entry squares.
    const boxMask = new Uint8Array(level.n);
    for (const c of room.cells) boxMask[c] = 1;
    const walkMask = new Uint8Array(level.n);
    for (const c of room.cells) walkMask[c] = 1;
    for (const e of room.entries) walkMask[e] = 1;
    const world = new PushWorld(level, boxMask, walkMask, room.goals);

    // Boxes outside this room stand still while it is solved; a room
    // that is not the last one only counts as solved when the player
    // can still walk out of it to the next room's entry squares.
    const outside = new Set([...globalBoxes].filter((b) => !roomBoxSet.has(b)));
    const accept = isLast
      ? null
      : (boxes, endPlayer) => {
          const blocked = new Set([...outside, ...boxes]);
          return walkPath(level, endPlayer, order[i + 1].entries, blocked) !== null;
        };

    // Walk to one of the room's entry squares and solve from there;
    // the first entry that leads to a full solution wins.
    let solvedRoom = null;
    for (const entry of room.entries) {
      const walk = walkPath(level, player, [entry], globalBoxes);
      if (walk === null) continue;
      const res = world.solve(room.boxes, entry, 300000, accept);
      if (res === null) continue;
      solvedRoom = { walkMoves: walk.moves, res };
      break;
    }
    if (solvedRoom === null) return null;

    moves += solvedRoom.walkMoves + solvedRoom.res.moves;
    player = solvedRoom.res.endPlayer;
    // The room's boxes now rest on its goals.
    for (const b of room.boxes) globalBoxes.delete(b);
    for (const gCell of room.goals) globalBoxes.add(gCell);
  }
  return moves;
}

// ---------------------------------------------------------------------
// The entry point: answer every instance in the input directory.
// A failure on one instance must never take the others down with it.
// ---------------------------------------------------------------------
function main() {
  // The set's wall budget is 60 s. A level not yet started when the
  // guard trips gets an empty (valid, scoreless) answer instead of
  // risking the run being killed mid-level and losing the answers
  // earlier levels already earned.
  const setStart = Date.now();
  for (const id of readdirSync(INPUT_DIR).sort()) {
    let moves = '';
    try {
      if (Date.now() - setStart > 50_000) throw new Error('set time guard');
      const level = parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
      moves = solveLevel(level);
    } catch {
      moves = '';
    }
    if (moves.length > 60000) moves = '';
    mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
    writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
  }
}

main();
