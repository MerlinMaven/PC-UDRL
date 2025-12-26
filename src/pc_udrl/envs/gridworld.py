import numpy as np
import numpy.typing as npt


class GridWorld:
    def __init__(self, size=5, seed=0, fixed_map=True):
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.fixed_map = fixed_map
        self.walls = []
        self.traps = []
        self._walls_set = set()
        self._traps_set = set()
        self.reset()

    def reset(self):
        self.agent = np.array([0, 0])
        self.goal = np.array([self.size - 1, self.size - 1])
        # Only place obstacles if not fixed or if it's the first reset (empty lists)
        if not self.fixed_map or not self.walls:
            self._place_obstacles()
        return self._obs()

    def step(self, action):
        y, x = int(self.agent[0]), int(self.agent[1])
        ny, nx = y, x
        if action == 0:
            ny = max(0, y - 1)
        elif action == 1:
            ny = min(self.size - 1, y + 1)
        elif action == 2:
            nx = max(0, x - 1)
        elif action == 3:
            nx = min(self.size - 1, x + 1)
        r = -1.0
        done = False
        if (ny, nx) in self._walls_set:
            r -= 0.5
        else:
            self.agent[0] = ny
            self.agent[1] = nx
        if (int(self.agent[0]), int(self.agent[1])) in self._traps_set:
            r = -10.0
            done = True
        if (self.agent == self.goal).all():
            r = 10.0
            done = True
        return self._obs(), r, done, {}

    def sample_action(self):
        return self.rng.randint(0, 4)

    @property
    def observation_space(self):
        return (2,)

    @property
    def action_space(self):
        return 4

    def _obs(self):
        return self.agent.astype(np.float32)

    def _place_obstacles(self):
        self.walls = []
        self.traps = []
        count_walls = max(0, self.size - 3)
        count_traps = max(0, self.size - 4)
        used = {(int(self.agent[0]), int(self.agent[1])), (int(self.goal[0]), int(self.goal[1]))}
        while len(self.walls) < count_walls:
            wy = int(self.rng.randint(0, self.size))
            wx = int(self.rng.randint(0, self.size))
            if (wy, wx) not in used:
                self.walls.append((wy, wx))
                used.add((wy, wx))
        while len(self.traps) < count_traps:
            ty = int(self.rng.randint(0, self.size))
            tx = int(self.rng.randint(0, self.size))
            if (ty, tx) not in used:
                self.traps.append((ty, tx))
                used.add((ty, tx))
        self._walls_set = set(self.walls)
        self._traps_set = set(self.traps)

    def render(self, mode: str = "rgb_array", cell_px: int = 32) -> npt.NDArray[np.uint8]:
        H = self.size * cell_px
        W = self.size * cell_px
        img = np.ones((H, W, 3), dtype=np.uint8) * 240
        for i in range(self.size + 1):
            y = i * cell_px
            img[y:y+1, :, :] = 220
            x = i * cell_px
            img[:, x:x+1, :] = 220
        for (wy, wx) in self.walls:
            y0 = wy * cell_px
            x0 = wx * cell_px
            img[y0:y0+cell_px, x0:x0+cell_px, :] = np.array([0, 0, 0], dtype=np.uint8)
        for (ty, tx) in self.traps:
            y0 = ty * cell_px
            x0 = tx * cell_px
            img[y0:y0+cell_px, x0:x0+cell_px, :] = np.array([220, 20, 60], dtype=np.uint8)
        gy0 = int(self.goal[0]) * cell_px
        gx0 = int(self.goal[1]) * cell_px
        img[gy0:gy0+cell_px, gx0:gx0+cell_px, :] = np.array([0, 200, 0], dtype=np.uint8)
        ay = int(self.agent[0])
        ax = int(self.agent[1])
        cy = ay * cell_px + cell_px // 2
        cx = ax * cell_px + cell_px // 2
        rad = cell_px // 3
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
        img[mask] = np.array([30, 144, 255], dtype=np.uint8)
        return img
