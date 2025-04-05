import pygame
import numpy as np
import pygame.gfxdraw
from scipy.integrate import ode

# Constants
G = 6.67430e-11  # Gravitational constant
M_sun = 1.989e30
AU = 1.496e11
YEAR = 365.25 * 24 * 3600
R_earth = 6.371e6  # Earth's radius in meters

# Pygame setup
WIDTH, HEIGHT = 1000, 1000
SCALE = WIDTH / (7 * AU)
CENTER = (WIDTH // 2, HEIGHT // 2)
CAM_WIDTH, CAM_HEIGHT = 200, 200
CAM_SCALE = 1e-6  # Zoom factor for satellite camera
CAM_POS = (WIDTH - CAM_WIDTH - 10, 10)  # Top-right corner

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

# Integration step
dt_vis = 3600 * 12
trail = []

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

# Create bodies
bodies = [
    HeavenlyBody("Sun", M_sun, 0, 0, 0, 0),
    HeavenlyBody("Earth", 5.972e24, AU, 0, 0, np.sqrt(G * M_sun / AU)),
    HeavenlyBody("Mars", 6.4171e25, 1.524 * AU, 0, 0, np.sqrt(G * M_sun / (1.524 * AU))),
]

# Place satellite in low Earth orbit
earth_x = bodies[1].state[0]
earth_y = bodies[1].state[1]
earth_vx = bodies[1].state[2]
earth_vy = bodies[1].state[3]

orbital_radius = R_earth + 400e5  # 400 km altitude
sat_x = earth_x + orbital_radius
sat_y = earth_y
sat_v = np.sqrt(G * bodies[1].mass / orbital_radius)
sat_vx = earth_vx
sat_vy = earth_vy + sat_v

satellite = Satellite("Satellite", 1000, sat_x, sat_y, sat_vx, sat_vy)
satellite.thrustX = 0  # meters per second^2
satellite.thrustY = 0

bodies.append(satellite)

masses = [body.mass for body in bodies]

def flatten_states(bodies):
    return np.concatenate([body.state for body in bodies])

def equations(t, state, masses):
    derivatives = []
    num_bodies = len(masses)
    for i in range(num_bodies):
        xi, yi, vxi, vyi = state[i*4:i*4+4]
        axi = ayi = 0
        for j in range(num_bodies):
            if i == j:
                continue
            xj, yj = state[j*4:j*4+2]
            mj = masses[j]
            dx = xj - xi
            dy = yj - yi
            r = np.sqrt(dx**2 + dy**2) + 1e-10
            axi += G * mj * dx / r**3
            ayi += G * mj * dy / r**3

        # Inject thrust for Satellite
        if bodies[i].name == "Satellite":
            axi += satellite.thrustX / satellite.mass
            ayi += satellite.thrustY / satellite.mass

        derivatives.extend([vxi, vyi, axi, ayi])
    return derivatives

# Initialize ODE solver
initial_state = flatten_states(bodies)
solver = ode(lambda t, y: equations(t, y, masses)).set_integrator('dop853')
solver.set_initial_value(initial_state, 0)

while running:
    screen.fill((0, 0, 0))

    font = pygame.font.SysFont(None, 24)
    text_surface = font.render(f"Time: {solver.t / (24 * 3600):.2f} days", True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))

    if solver.successful():
        solver.integrate(solver.t + dt_vis)

        new_state = solver.y
        for i, body in enumerate(bodies):
            body.update_state(new_state[i*4:(i+1)*4])

        satellite = next(b for b in bodies if b.name == "Satellite")
        sat_x, sat_y = satellite.get_position()

        for body in bodies:
            x, y = body.get_position()
            screen_x = int(CENTER[0] + x * SCALE)
            screen_y = int(CENTER[1] - y * SCALE)

            if body.name == "Sun":
                pygame.draw.circle(screen, (255, 255, 0), (screen_x, screen_y), 10)
            elif body.name == "Earth":
                pygame.draw.circle(screen, (0, 0, 255), (screen_x, screen_y), 5)
                text = f"Earth - X: {x:.2e}, Y: {y:.2e}"
                screen.blit(font.render(text, True, (255, 255, 255)), (10, 30))
            elif body.name == "Mars":
                pygame.draw.circle(screen, (255, 100, 100), (screen_x, screen_y), 5)
                text = f"Mars - X: {x:.2e}, Y: {y:.2e}"
                screen.blit(font.render(text, True, (255, 255, 255)), (10, 50))
            elif body.name == "Satellite":
                pygame.draw.circle(screen, (0, 255, 0), (screen_x, screen_y), 3)

        # Draw satellite camera view
        cam_surface = pygame.Surface((CAM_WIDTH, CAM_HEIGHT))
        cam_surface.fill((20, 20, 20))
        for body in bodies:
            x, y = body.get_position()
            rel_x = (x - sat_x) * CAM_SCALE + CAM_WIDTH // 2
            rel_y = (sat_y - y) * CAM_SCALE + CAM_HEIGHT // 2
            if body.name == "Satellite":
                pygame.draw.circle(cam_surface, (0, 255, 0), (int(rel_x), int(rel_y)), 4)
            elif body.name == "Earth":
                pygame.draw.circle(cam_surface, (0, 0, 255), (int(rel_x), int(rel_y)), 5)
            elif body.name == "Sun":
                pygame.draw.circle(cam_surface, (255, 255, 0), (int(rel_x), int(rel_y)), 5)
        pygame.draw.rect(screen, (255, 255, 255), (*CAM_POS, CAM_WIDTH, CAM_HEIGHT), 1)
        screen.blit(cam_surface, CAM_POS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(60)

pygame.quit()