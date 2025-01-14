import pygame
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import random
import math
import os

# For the pop-out logs
import tkinter as tk
import tkinter.scrolledtext as st


pygame.init()

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption(
    "Multi-Pawn Quantum Maze (3 Qubits + Shared RL + Maze Generation)"
)

LEFT_PANEL_WIDTH = 700
RIGHT_PANEL_WIDTH = 300

LEFT_PANEL_RECT = pygame.Rect(0, 0, LEFT_PANEL_WIDTH, WINDOW_HEIGHT)
RIGHT_PANEL_RECT = pygame.Rect(
    LEFT_PANEL_WIDTH, 0, RIGHT_PANEL_WIDTH, WINDOW_HEIGHT
)

CHART_HEIGHT = 250
LOG_HEIGHT = 300
CHART_RECT = pygame.Rect(LEFT_PANEL_WIDTH, 0, RIGHT_PANEL_WIDTH, CHART_HEIGHT)
LOG_RECT = pygame.Rect(
    LEFT_PANEL_WIDTH, CHART_HEIGHT, RIGHT_PANEL_WIDTH, LOG_HEIGHT
)

BUTTON_HEIGHT = 50
BUTTON_RECT = pygame.Rect(
    LEFT_PANEL_WIDTH,
    CHART_HEIGHT + LOG_HEIGHT,
    RIGHT_PANEL_WIDTH,
    BUTTON_HEIGHT,
)

font = pygame.font.SysFont("Arial", 20)
log_font = pygame.font.SysFont("Consolas", 16)
clock = pygame.time.Clock()

# Comment out to disable anomaly detection (for speed)
torch.autograd.set_detect_anomaly(True)


root = tk.Tk()
root.withdraw()
log_window = None
log_text_area = None


def create_log_window():
    global log_window, log_text_area
    if log_window and tk.Toplevel.winfo_exists(log_window):
        return
    log_window = tk.Toplevel(root)
    log_window.title("Action Logs - Pop Out")
    log_text_area = st.ScrolledText(log_window, width=60, height=25)
    log_text_area.pack(fill=tk.BOTH, expand=True)


def log_to_tkinter(msg):
    global log_text_area
    if log_text_area:
        log_text_area.insert(tk.END, msg + "\n")
        log_text_area.see(tk.END)


# RANDOM MAZE GENERATION


def generate_random_maze(rows, cols):
    """
    DFS-based random maze generation.
    '1' => wall, '0' => open path.
    Then place 'S' (start) near top-left, 'E' near bottom-right.
    """
    maze = [["1" for _ in range(cols)] for _ in range(rows)]

    # Carve with DFS
    start_r = random.randrange(rows)
    start_c = random.randrange(cols)
    maze[start_r][start_c] = "0"
    stack = [(start_r, start_c)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        r, c = stack[-1]
        neighbors = []
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + 2 * dr, c + 2 * dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] == "1":
                    neighbors.append((dr, dc))
        if not neighbors:
            stack.pop()
        else:
            dr, dc = random.choice(neighbors)
            maze[r + dr][c + dc] = "0"
            maze[r + 2 * dr][c + 2 * dc] = "0"
            stack.append((r + 2 * dr, c + 2 * dc))

    # Place 'S'
    placed_S = False
    for rr in range(rows):
        for cc in range(cols):
            if maze[rr][cc] == "0":
                maze[rr][cc] = "S"
                placed_S = True
                break
        if placed_S:
            break

    # Place 'E'
    placed_E = False
    for rr in range(rows - 1, -1, -1):
        for cc in range(cols - 1, -1, -1):
            if maze[rr][cc] == "0":
                maze[rr][cc] = "E"
                placed_E = True
                break
        if placed_E:
            break

    return ["".join(row) for row in maze]


# NEURAL NETWORK (3 QUBITS - DEFAULT)


class GateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 10  # row, col, 8 one-hot for last outcome
        self.hidden_dim = 16
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 6),  # 6 => 2 logits for q0, q1, q2
        )

    def forward(self, state_tensor):
        logits = self.net(state_tensor)
        # split: 2 for q0, 2 for q1, 2 for q2
        logits_q0 = logits[0:2]
        logits_q1 = logits[2:4]
        logits_q2 = logits[4:6]
        probs_q0 = torch.softmax(logits_q0, dim=0)
        probs_q1 = torch.softmax(logits_q1, dim=0)
        probs_q2 = torch.softmax(logits_q2, dim=0)
        return probs_q0, probs_q1, probs_q2


# QUANTUM / QISKIT SETUP
simulator = AerSimulator()
shots_per_step = 1

direction_counts = {
    bits: 0 for bits in ["000", "001", "010", "011", "100", "101", "110", "111"]
}


def build_circuit(gates):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    for i, g in enumerate(gates):
        if g == "H":
            qc.h(i)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def measure_3qubits(gates):
    qc = build_circuit(gates)
    t_qc = transpile(qc, simulator)
    job = simulator.run(t_qc, shots=shots_per_step)
    result = job.result()
    counts = result.get_counts()
    outcome = list(counts.keys())[0]  # e.g. '010'
    return outcome


dir_map = {
    "000": (-1, 0),
    "001": (-1, +1),
    "010": (0, +1),
    "011": (+1, +1),
    "100": (+1, 0),
    "101": (+1, -1),
    "110": (0, -1),
    "111": (-1, -1),
}


# GLOBALS / Maze placeholders
MAZE_MAP = []
ROWS = 0
COLS = 0
TILE_SIZE = 40

start_positions = []
exit_pos = None

visited_count = []  # track how many times each cell was visited


def is_wall(r, c):
    if r < 0 or r >= ROWS or c < 0 or c >= COLS:
        return True
    return MAZE_MAP[r][c] == "1"


def is_exit(r, c):
    return (r, c) == exit_pos


def is_corner(r, c):
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, +1),
        (+1, -1),
        (+1, +1),
    ]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if not is_wall(nr, nc):
            return False
    return True


# MULTI-PAWN SETUP
NUM_PAWNS = 3
pawn_states = []


def initialize_pawn_states():
    global pawn_states
    pawn_states = []
    for i in range(NUM_PAWNS):
        sp = start_positions[i % len(start_positions)]
        pawn_states.append(
            {
                "r": sp[0],
                "c": sp[1],
                "found_exit": False,
                "last_outcome": None,
                "episode_log_probs": [],
                "episode_rewards": [],
                "step_count": 0,
                "done": False,
                # short-term memory for penalizing re-visits
                "recent_positions": deque([], maxlen=20),
            }
        )


# RL

gate_net = GateNet()
optimizer = optim.Adam(gate_net.parameters(), lr=1e-3)
gamma = 0.99
max_steps_per_episode = 1000
EPSILON = 0.05  # 10% random


def make_state(r, c, last_outcome):
    oh = [0] * 8
    if last_outcome is not None:
        idx_map = {
            "000": 0,
            "001": 1,
            "010": 2,
            "011": 3,
            "100": 4,
            "101": 5,
            "110": 6,
            "111": 7,
        }
        oh[idx_map[last_outcome]] = 1
    R = float(min(r, 10))
    C = float(min(c, 10))
    return torch.tensor([R, C] + oh, dtype=torch.float)


def manhattan_dist(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)


# do_one_step_for_pawn


def do_one_step_for_pawn(pawn_idx):
    """
    Revised reward system to "feel good" going the right direction:
      - corner => -2
      - base step cost => -0.05
      - re-visit recent => -0.3
      - distance decreases => +0.5
      - distance increases => -0.2
      - wall => -2
      - exit => +50
    """
    p = pawn_states[pawn_idx]
    if p["done"]:
        return

    # corner => big penalty, end
    if is_corner(p["r"], p["c"]):
        p["episode_log_probs"].append(torch.tensor(0.0))
        p["episode_rewards"].append(-2.0)  # corner penalty
        msg = f"Pawn {pawn_idx} corner => -2, end"
        log_action(msg)
        log_to_tkinter(msg)
        p["done"] = True
        return

    old_r, old_c = p["r"], p["c"]
    p["step_count"] += 1

    # Build state for GateNet
    s_tensor = make_state(p["r"], p["c"], p["last_outcome"])
    q0, q1, q2 = gate_net(s_tensor)

    # EPSILON exploration
    if random.random() < EPSILON:
        q0idx = random.randint(0, 1)
        q1idx = random.randint(0, 1)
        q2idx = random.randint(0, 1)
    else:
        q0idx = torch.multinomial(q0, 1).item()
        q1idx = torch.multinomial(q1, 1).item()
        q2idx = torch.multinomial(q2, 1).item()

    logprob = (
        torch.log(q0[q0idx]) + torch.log(q1[q1idx]) + torch.log(q2[q2idx])
    ).clone()

    gates = [("H" if i == 0 else "I") for i in (q0idx, q1idx, q2idx)]
    outcome = measure_3qubits(gates)
    direction_counts[outcome] += 1
    (dr, dc) = dir_map[outcome]
    nr, nc = old_r + dr, old_c + dc

    # base step cost
    reward = -0.05

    # short-term memory re-visit
    if (nr, nc) in p["recent_positions"]:
        reward -= 0.3

    # wall check
    if is_wall(nr, nc):
        reward -= 2.0  # bigger penalty for wall
        msg = (
            f"[Pawn{pawn_idx}] Step={p['step_count']}, gates={gates}, "
            f"outcome={outcome}, WALL => rew={reward:.2f}"
        )
        log_action(msg)
        log_to_tkinter(msg)
    else:
        old_dist = manhattan_dist(old_r, old_c, exit_pos[0], exit_pos[1])
        new_dist = manhattan_dist(nr, nc, exit_pos[0], exit_pos[1])

        # distance-based shaping
        if new_dist < old_dist:
            reward += 0.5  # feel good going closer
        elif new_dist > old_dist:
            reward -= 0.2  # penalize going away

        # move
        p["r"], p["c"] = nr, nc
        p["recent_positions"].append((p["r"], p["c"]))

        # also track breadcrumbs
        visited_count[p["r"]][p["c"]] += 1

        msg = (
            f"[Pawn{pawn_idx}] Step={p['step_count']}, gates={gates}, outcome={outcome}, "
            f"pos=({p['r']},{p['c']}), rew={reward:.2f}"
        )
        log_action(msg)
        log_to_tkinter(msg)

        # exit check
        if (p["r"], p["c"]) == exit_pos:
            reward += 50.0
            p["found_exit"] = True
            msg = f"[Pawn{pawn_idx}] Reached EXIT => +50, done"
            log_action(msg)
            log_to_tkinter(msg)

    # store logs
    p["episode_log_probs"].append(logprob)
    p["episode_rewards"].append(reward)
    p["last_outcome"] = outcome

    # done if exit or step limit
    if p["found_exit"] or p["step_count"] >= max_steps_per_episode:
        p["done"] = True


def batch_finish_and_update():
    """
    Weighted by the best pawn's raw (undiscounted) returns => alpha
    Then do a single backward pass.
    """
    total_policy_loss = None

    # compute raw returns for each pawn
    all_returns = [sum(p["episode_rewards"]) for p in pawn_states]
    r_max = max(all_returns) if all_returns else 1e-8

    for i, p in enumerate(pawn_states):
        ep_rewards = p["episode_rewards"]
        if len(ep_rewards) == 0:
            continue

        # discounted returns
        G = 0
        returns = []
        for r in reversed(ep_rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float)
        # standardize
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # build policy loss
        pawn_loss = torch.tensor(0.0)
        for lp, Rt in zip(p["episode_log_probs"], returns_t):
            pawn_loss += -lp.clone() * Rt.clone()

        # scale factor alpha
        raw_return = all_returns[i]
        scaled_return = max(0, raw_return)  # ignoring negative
        alpha = scaled_return / (r_max if r_max > 0 else 1e-8)
        pawn_loss *= alpha

        if total_policy_loss is None:
            total_policy_loss = pawn_loss
        else:
            total_policy_loss = total_policy_loss + pawn_loss

    if total_policy_loss is None:
        return

    optimizer.zero_grad()
    total_policy_loss.backward()
    optimizer.step()

    # save model for next time
    torch.save(gate_net.state_dict(), "model_checkpoint.pth")

    msg = f"BATCH UPDATE done. Weighted by best pawn. Returns: {all_returns}, r_max={r_max:.2f}"
    log_action(msg)
    log_to_tkinter(msg)

    # reset for next batch
    for p in pawn_states:
        p["episode_log_probs"].clear()
        p["episode_rewards"].clear()
        p["done"] = False
        p["found_exit"] = False
        p["step_count"] = 0
        p["recent_positions"].clear()
        p["last_outcome"] = None
        # re-init position
        sp = start_positions[0]
        p["r"], p["c"] = sp[0], sp[1]


# DRAWING & LOG

action_log = deque()
MAX_LOG_LINES = 200


def log_action(msg):
    if len(action_log) >= MAX_LOG_LINES:
        action_log.popleft()
    action_log.append(msg)
    print(msg)


def draw_log(surface):
    surface.fill((20, 20, 20))
    title = font.render("Action Log (auto-scroll)", True, (255, 255, 255))
    surface.blit(title, (10, 10))
    y_offset = surface.get_height() - 20
    line_height = 20
    lines_to_show = min(
        len(action_log), (surface.get_height() - 40) // line_height
    )
    start_idx = len(action_log) - lines_to_show

    for i in reversed(range(start_idx, len(action_log))):
        txt = action_log[i]
        line_surf = log_font.render(txt, True, (200, 200, 200))
        surface.blit(line_surf, (10, y_offset))
        y_offset -= line_height


def draw_chart(surface):
    surface.fill((30, 30, 30))
    title = font.render("3-Qubit Usage", True, (255, 255, 255))
    surface.blit(title, (10, 10))

    keys_order = ["000", "001", "010", "011", "100", "101", "110", "111"]
    total_moves = sum(direction_counts.values())
    if total_moves == 0:
        total_moves = 1
    max_count = max(direction_counts.values()) if direction_counts else 1

    chart_x = 20
    chart_y = 50
    chart_h = 150
    chart_w = surface.get_width() - 40
    bar_w = chart_w // len(keys_order)

    for i, bits in enumerate(keys_order):
        cnt = direction_counts[bits]
        bar_h = int((cnt / max_count) * chart_h)
        bx = chart_x + i * bar_w
        by = chart_y + (chart_h - bar_h)
        pygame.draw.rect(surface, (100, 180, 100), (bx, by, bar_w - 10, bar_h))
        lbl_surf = font.render(bits, True, (255, 255, 255))
        surface.blit(lbl_surf, (bx, chart_y + chart_h + 5))
        cnt_surf = font.render(str(cnt), True, (255, 255, 255))
        surface.blit(cnt_surf, (bx, by - 20))


SOLUTION_PATH = []


def draw_pawns_maze(surface):
    surface.fill((0, 0, 0))
    for rr in range(ROWS):
        for cc in range(COLS):
            tile = MAZE_MAP[rr][cc]
            px, py = cc * TILE_SIZE, rr * TILE_SIZE

            # Base color: walls => dark gray, open => lighter gray
            if tile == "1":
                base_color = (40, 40, 40)
            else:
                base_color = (90, 90, 90)

            pygame.draw.rect(
                surface, base_color, (px, py, TILE_SIZE, TILE_SIZE)
            )

            # Mark S/E
            if tile == "S":
                pygame.draw.circle(
                    surface,
                    (0, 255, 0),
                    (px + TILE_SIZE // 2, py + TILE_SIZE // 2),
                    6,
                )
            elif tile == "E":
                pygame.draw.circle(
                    surface,
                    (255, 0, 0),
                    (px + TILE_SIZE // 2, py + TILE_SIZE // 2),
                    6,
                )

            vcount = visited_count[rr][cc]
            if vcount > 0 and tile != "1":
                # max, e.g. 10 => full bright
                intensity = min(vcount, 10)
                green_val = 25 * intensity
                if green_val > 255:
                    green_val = 255
                pygame.draw.rect(
                    surface, (0, green_val, 0), (px, py, TILE_SIZE, TILE_SIZE)
                )

    if len(SOLUTION_PATH) > 1:
        path_color = (0, 255, 255)
        for i in range(len(SOLUTION_PATH) - 1):
            (r1, c1) = SOLUTION_PATH[i]
            (r2, c2) = SOLUTION_PATH[i + 1]
            x1 = c1 * TILE_SIZE + TILE_SIZE // 2
            y1 = r1 * TILE_SIZE + TILE_SIZE // 2
            x2 = c2 * TILE_SIZE + TILE_SIZE // 2
            y2 = r2 * TILE_SIZE + TILE_SIZE // 2
            pygame.draw.line(surface, path_color, (x1, y1), (x2, y2), 3)

    # Finally, draw pawns
    colors = [(0, 200, 200), (200, 200, 0), (200, 0, 200), (200, 100, 0)]
    for i, p in enumerate(pawn_states):
        col = colors[i % len(colors)]
        if p["found_exit"]:
            col = (255, 215, 0)
        px, py = p["c"] * TILE_SIZE, p["r"] * TILE_SIZE
        pygame.draw.rect(
            surface, col, (px + 5, py + 5, TILE_SIZE - 10, TILE_SIZE - 10)
        )


def find_solution_path():
    """
    BFS from 'S' to 'E'. Return a list of (r,c) coords if found, else [].
    """
    from collections import deque

    s_coord = None
    e_coord = None
    for rr in range(ROWS):
        for cc in range(COLS):
            if MAZE_MAP[rr][cc] == "S":
                s_coord = (rr, cc)
            elif MAZE_MAP[rr][cc] == "E":
                e_coord = (rr, cc)
    if not s_coord or not e_coord:
        return []
    queue = deque([s_coord])
    visited = set([s_coord])
    parent = dict()
    parent[s_coord] = None
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        if (r, c) == e_coord:
            path = []
            cur = e_coord
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                if MAZE_MAP[nr][nc] != "1" and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
    return []


show_keybinds = False


def draw_keybinds_overlay(surface):
    overlay = pygame.Surface((LEFT_PANEL_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))
    lines = [
        "[Tab] Toggle Keybinds/Instructions",
        "[A]   Toggle Auto-step",
        "[Space] Single step if auto-step off",
        "[P]   Pop out logs window",
        "[ESC] Quit",
    ]
    y = 50
    for line in lines:
        surf = font.render(line, True, (255, 255, 255))
        surface.blit(surf, (50, y))
        y += 30


AUTO_STEP = True
maze_generation_count = 0


def main():
    global AUTO_STEP, show_keybinds
    global MAZE_MAP, ROWS, COLS, start_positions, exit_pos, visited_count
    global maze_generation_count, SOLUTION_PATH

    # If a model checkpoint exists, load it
    if os.path.exists("model_checkpoint.pth"):
        gate_net.load_state_dict(torch.load("model_checkpoint.pth"))
        print("Loaded existing policy from model_checkpoint.pth")

    # Generate a random maze
    maze_generation_count += 1
    random_maze = generate_random_maze(15, 15)
    MAZE_MAP = random_maze
    ROWS = len(MAZE_MAP)
    COLS = len(MAZE_MAP[0])
    visited_count = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    # BFS solution path (purely for visual reference)
    SOLUTION_PATH = find_solution_path()
    print(f"Solution path length={len(SOLUTION_PATH)}")
    print(f"Generated Maze #{maze_generation_count} (size {ROWS}x{COLS})")

    # parse S, E
    start_positions.clear()
    exit_pos = None
    for rr in range(ROWS):
        for cc in range(COLS):
            if MAZE_MAP[rr][cc] == "S":
                start_positions.append((rr, cc))
            elif MAZE_MAP[rr][cc] == "E":
                exit_pos = (rr, cc)

    # init pawns
    initialize_pawn_states()

    frame_count = 0
    running = True
    while running:
        clock.tick(6)  # ~6 FPS
        frame_count += 1

        # step pawns every other frame (example)
        if frame_count % 2 == 0:
            for i in range(len(pawn_states)):
                do_one_step_for_pawn(i)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_a:
                    AUTO_STEP = not AUTO_STEP
                    log_action(f"AUTO_STEP => {AUTO_STEP}")
                    log_to_tkinter(f"AUTO_STEP => {AUTO_STEP}")
                elif event.key == pygame.K_SPACE:
                    if not AUTO_STEP:
                        for i in range(len(pawn_states)):
                            do_one_step_for_pawn(i)
                elif event.key == pygame.K_TAB:
                    show_keybinds = not show_keybinds
                elif event.key == pygame.K_p:
                    create_log_window()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if BUTTON_RECT.collidepoint(mx, my):
                    create_log_window()

        # auto-step if needed
        if AUTO_STEP:
            for i in range(len(pawn_states)):
                do_one_step_for_pawn(i)

        # batch update if all pawns done
        all_done = all(p["done"] for p in pawn_states)
        if all_done:
            batch_finish_and_update()

        # draw
        maze_surf = pygame.Surface((LEFT_PANEL_WIDTH, WINDOW_HEIGHT))
        draw_pawns_maze(maze_surf)
        if show_keybinds:
            draw_keybinds_overlay(maze_surf)
        screen.blit(maze_surf, (0, 0))

        chart_surf = pygame.Surface((RIGHT_PANEL_WIDTH, CHART_HEIGHT))
        draw_chart(chart_surf)
        screen.blit(chart_surf, (CHART_RECT.x, CHART_RECT.y))

        log_surf = pygame.Surface((RIGHT_PANEL_WIDTH, LOG_HEIGHT))
        draw_log(log_surf)
        screen.blit(log_surf, (LOG_RECT.x, LOG_RECT.y))

        # pop-out logs button
        button_surf = pygame.Surface((RIGHT_PANEL_WIDTH, BUTTON_HEIGHT))
        button_surf.fill((50, 50, 50))
        text_surf = font.render(
            "Pop Out Logs (Click 'P')", True, (255, 255, 255)
        )
        tx = (button_surf.get_width() - text_surf.get_width()) // 2
        ty = (button_surf.get_height() - text_surf.get_height()) // 2
        button_surf.blit(text_surf, (tx, ty))
        screen.blit(button_surf, (BUTTON_RECT.x, BUTTON_RECT.y))

        root.update()  # keep Tkinter UI responsive
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
