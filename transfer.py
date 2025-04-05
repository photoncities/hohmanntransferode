import pygame
import numpy as np
import pygame.gfxdraw
from scipy.integrate import ode

# Constants
G = 6.67430e-11
M_sun = 1.989e30
AU = 1.496e11
YEAR = 365.25 * 24 * 3600
R_earth = 6.371e6

WIDTH, HEIGHT = 1000, 1000
SCALE = WIDTH / (7 * AU)
CENTER = (WIDTH // 2, HEIGHT // 2)
CAM_WIDTH, CAM_HEIGHT = 200, 200
CAM_SCALE = 1e-7
CAM_POS = (WIDTH - CAM_WIDTH - 10, 10)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True
frame_count = 0

fast_dt = 3600 * 12
slow_dt = 360
dt_vis = fast_dt
speed_toggle = False

class HeavenlyBody:
    def __init__(self, name, mass, x, y, vx, vy):
        self.name = name
        self.mass = mass
        self.state = np.array([x, y, vx, vy], dtype=np.float64)

    def get_position(self):
        return self.state[0], self.state[1]

    def get_velocity(self):
        return self.state[2], self.state[3]

    def update_state(self, new_state):
        self.state = np.array(new_state, dtype=np.float64)

class Satellite(HeavenlyBody):
    def __init__(self, name, mass, x, y, vx, vy):
        super().__init__(name, mass, x, y, vx, vy)
        self.thrustX = 0
        self.thrustY = 0

bodies = [
    HeavenlyBody("Sun", M_sun, 0, 0, 0, 0),
    HeavenlyBody("Earth", 5.972e24, AU, 0, 0, np.sqrt(G * M_sun / AU)),
    HeavenlyBody("Mars", 6.4171e25, 1.524 * AU, 0, 0, np.sqrt(G * M_sun / (1.524 * AU))),
]

earth = bodies[1]
earth_x, earth_y = earth.state[0:2]
earth_vx, earth_vy = earth.state[2:4]
orbital_radius = R_earth + 400e5
sat_x = earth_x + orbital_radius
sat_y = earth_y
sat_v = np.sqrt(G * earth.mass / orbital_radius)
sat_vx = earth_vx
sat_vy = earth_vy + sat_v

satellite = Satellite("Satellite", 1000, sat_x, sat_y, sat_vx, sat_vy)
bodies.append(satellite)

masses = [b.mass for b in bodies]

def flatten_states(bodies):
    return np.concatenate([body.state for body in bodies])

def equations(t, state, masses):
    derivatives = []
    for i in range(len(masses)):
        xi, yi, vxi, vyi = state[i * 4:i * 4 + 4]
        axi = ayi = 0
        for j in range(len(masses)):
            if i == j:
                continue
            xj, yj = state[j * 4:j * 4 + 2]
            mj = masses[j]
            dx, dy = xj - xi, yj - yi
            r = np.sqrt(dx**2 + dy**2) + 1e-10
            axi += G * mj * dx / r**3
            ayi += G * mj * dy / r**3
        if bodies[i].name == "Satellite":
            axi += satellite.thrustX / satellite.mass
            ayi += satellite.thrustY / satellite.mass
        derivatives.extend([vxi, vyi, axi, ayi])
    return derivatives

solver = ode(lambda t, y: equations(t, y, masses)).set_integrator('dop853')
solver.set_initial_value(flatten_states(bodies), 0)

def predict_trajectory(sat, steps=5, step_size=3600):
    x, y = sat.get_position()
    vx, vy = sat.get_velocity()
    points = []
    for _ in range(steps):
        ax = ay = 0
        for b in bodies:
            if b is sat:
                continue
            bx, by = b.get_position()
            dx = bx - x
            dy = by - y
            r = np.sqrt(dx**2 + dy**2) + 1e-10
            a = G * b.mass / r**2
            ax += a * dx / r
            ay += a * dy / r
        ax += sat.thrustX / sat.mass
        ay += sat.thrustY / sat.mass
        vx += ax * step_size
        vy += ay * step_size
        x += vx * step_size
        y += vy * step_size
        sx = int(CENTER[0] + x * SCALE)
        sy = int(CENTER[1] - y * SCALE)
        points.append((sx, sy, x, y))
    return points

while running:
    # Update thrust direction toward Earth
    sat_x, sat_y = satellite.get_position()
    earth_x, earth_y = earth.get_position()
    dx = earth_x - sat_x
    dy = earth_y - sat_y
    dist = np.sqrt(dx**2 + dy**2)
    if dist > 0:
        dir_unit = np.array([dx, dy]) / dist
        satellite.thrustX = 10 * dir_unit[0]
        satellite.thrustY = 10 * dir_unit[1]

    screen.fill((0, 0, 0))
    font = pygame.font.SysFont(None, 24)
    screen.blit(font.render(f"Time: {solver.t:.2f} s", True, (255, 255, 255)), (10, 10))

    if solver.successful():
        solver.integrate(solver.t + dt_vis)
        new_state = solver.y
        for i, b in enumerate(bodies):
            b.update_state(new_state[i * 4:(i + 1) * 4])

        if frame_count % 5 == 0:
            pred_trail = predict_trajectory(satellite, 5, 3600 * 4)

        for b in bodies:
            x, y = b.get_position()
            sx = int(CENTER[0] + x * SCALE)
            sy = int(CENTER[1] - y * SCALE)
            color = (255, 255, 0) if b.name == "Sun" else (0, 0, 255) if b.name == "Earth" else (255, 100, 100) if b.name == "Mars" else (0, 255, 0)
            pygame.draw.circle(screen, color, (sx, sy), 5 if b.name != "Sun" else 10)
            if b.name == "Satellite":
                tx, ty = satellite.thrustX, satellite.thrustY
                mag = np.sqrt(tx**2 + ty**2)
                if mag > 0:
                    tx /= mag
                    ty /= mag
                    tip = (sx + tx * 12, sy - ty * 12)
                    left = (sx - ty * 5, sy - tx * 5)
                    right = (sx + ty * 5, sy + tx * 5)
                    pygame.draw.polygon(screen, (0, 255, 255), [tip, left, right])

        for pt in pred_trail:
            pygame.draw.circle(screen, (200, 200, 200), pt[:2], 1)

        # Camera rendering
        cam_surface = pygame.Surface((CAM_WIDTH, CAM_HEIGHT))
        cam_surface.fill((20, 20, 20))
        for b in bodies:
            bx, by = b.get_position()
            cx = (bx - sat_x) * CAM_SCALE + CAM_WIDTH // 2
            cy = (by - sat_y) * CAM_SCALE + CAM_HEIGHT // 2
            color = (255, 255, 0) if b.name == "Sun" else (0, 0, 255) if b.name == "Earth" else (255, 100, 100) if b.name == "Mars" else (0, 255, 0)
            pygame.draw.circle(cam_surface, color, (int(cx), int(cy)), 4 if b.name != "Sun" else 6)
            if b.name == "Satellite":
                tx, ty = satellite.thrustX, satellite.thrustY
                mag = np.sqrt(tx**2 + ty**2)
                if mag > 0:
                    tx /= mag
                    ty /= mag
                    tip = (cx + tx * 12, cy - ty * 12)
                    left = (cx - ty * 5, cy - tx * 5)
                    right = (cx + ty * 5, cy + tx * 5)
                    pygame.draw.polygon(cam_surface, (0, 255, 255), [tip, left, right])


       
        for pt in pred_trail:
            px = (pt[2] - sat_x) * CAM_SCALE + CAM_WIDTH // 2
            py = (pt[3] - sat_y) * CAM_SCALE + CAM_HEIGHT // 2
            pygame.draw.circle(cam_surface, (0, 200, 0), (int(px), int(py)), 2)

        pygame.draw.rect(screen, (255, 255, 255), (*CAM_POS, CAM_WIDTH, CAM_HEIGHT), 1)
        screen.blit(cam_surface, CAM_POS)
        # Re-center cam on screen
        cam_center_x = CAM_POS[0] + CAM_WIDTH // 2
        cam_center_y = CAM_POS[1] + CAM_HEIGHT // 2
        pygame.draw.circle(screen, (0, 255, 255), (cam_center_x, cam_center_y), 3)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
            speed_toggle = not speed_toggle
            dt_vis = slow_dt if speed_toggle else fast_dt

    pygame.display.flip()
    clock.tick(60)
    frame_count += 1

pygame.quit()
