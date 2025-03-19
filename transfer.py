import pygame
import numpy as np
import pygame.gfxdraw
from scipy.integrate import ode

# Constants
G = 6.67430e-11  # Gravitational constant, m^3/kg/s^2
M_sun = 1.989e30  # Mass of the Sun, kg
AU = 1.496e11  # Astronomical unit, meters
YEAR = 365.25 * 24 * 3600  # One year in seconds

# Mars
M_mars = 6.4171e25  # Mass of Mars, kg
distance_mars = 1.524 * AU  # Average distance from the Sun, meters
initial_speed_mars = np.sqrt(G * M_sun * ((2 / distance_mars) - (1 / distance_mars)))  # m/s



# Earth orbital parameters
perihelion = AU # Closest approach to the Sun (same as AU for circular orbit)
aphelion = AU    # Farthest point from the Sun (same as AU for circular orbit)
initial_speed = np.sqrt(G * M_sun * ((2 / perihelion) - (1 / AU)))  # m/s

# Initial conditions (x, y, vx, vy)
state0 = np.array([perihelion, 0, 0, initial_speed, distance_mars, 0, 0, initial_speed_mars], dtype=np.float64)

# Pygame setup
WIDTH, HEIGHT = 1000, 1000
SCALE = WIDTH / (7 * AU)  # Scale factor for visualization
CENTER = (WIDTH // 2, HEIGHT // 2)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

# Integration step size
dt_vis = 3600 * 12  # 6-hour steps
trail = []

def equations(t, state):
    """Compute derivatives for the system (x, y, vx, vy, x_mars, y_mars, vx_mars, vy_mars)."""
    x, y, vx, vy, x_mars, y_mars, vx_mars, vy_mars = state
    r_earth = np.sqrt(x**2 + y**2)
    r_mars = np.sqrt(x_mars**2 + y_mars**2)

    # Gravitational acceleration for Earth
    ax = -G * M_sun * x / r_earth**3
    ay = -G * M_sun * y / r_earth**3
    r_earth_mars = np.sqrt((x - x_mars)**2 + (y - y_mars)**2)
    ax += G * M_mars * (x_mars - x) / r_earth_mars**3
    ay += G * M_mars * (y_mars - y) / r_earth_mars**3


    # Gravitational acceleration for Mars
    ax_mars = -G * M_sun * x_mars / r_mars**3
    ay_mars = -G * M_sun * y_mars / r_mars**3
    # Gravitational influence of Mars on Earth
    # Gravitational influence of Earth on Mars
    ax_mars += G * M_mars * (x - x_mars) / r_earth_mars**3
    ay_mars += G * M_mars * (y - y_mars) / r_earth_mars**3

    return [vx, vy, ax, ay, vx_mars, vy_mars, ax_mars, ay_mars]

# Initialize ODE solver
solver = ode(equations).set_integrator('dop853')
solver.set_initial_value(state0, 0)  # Initial state at t=0

while running:
    screen.fill((0, 0, 0))  # Clear screen

    # Draw text on screen
    font = pygame.font.SysFont(None, 24)
    text_surface = font.render(f"Time: {solver.t:.2f} s", True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))
    
    text_surface = font.render(f"Earth - X: {solver.y[0]:.2e} m, Y: {solver.y[1]:.2e} m, VX: {solver.y[2]:.2e} m/s, VY: {solver.y[3]:.2e} m/s", True, (255, 255, 255))
    screen.blit(text_surface, (10, 30))
    
    text_surface = font.render(f"Mars - X: {solver.y[4]:.2e} m, Y: {solver.y[5]:.2e} m, VX: {solver.y[6]:.2e} m/s, VY: {solver.y[7]:.2e} m/s", True, (255, 255, 255))
    screen.blit(text_surface, (10, 50))
    # Advance the solver by dt_vis
    if solver.successful():


        
        
        solver.integrate(solver.t + dt_vis)
        x, y, vx, vy, x_mars, y_mars, vx_mars, vy_mars = solver.y  # Extract updated values

        # Convert to screen coordinates
        screen_x, screen_y = int(CENTER[0] + (x * SCALE)), int(CENTER[1] - (y * SCALE))

        trail.append((screen_x, screen_y))
        # for pos in trail:
        #     pygame.draw.circle(screen, (0, 0, 255), pos, 2)
        # Draw Sun
        pygame.draw.circle(screen, (255, 255, 0), CENTER, 10)

        # Draw Earth
        distance = np.sqrt((screen_x - CENTER[0])**2 + (screen_y - CENTER[1])**2)
        pygame.gfxdraw.circle(screen, CENTER[0], CENTER[1], int(distance), (0, 0, 200))
        
        pygame.draw.circle(screen, (0, 0, 255), (screen_x, screen_y), 5)

        # Determine if Earth's orbit is elliptical or circular
  
       
        # Convert to screen coordinates for Mars
        screen_x_mars, screen_y_mars = int(CENTER[0] + (x_mars * SCALE)), int(CENTER[1] - (y_mars * SCALE))
        
   
        distance_mars = np.sqrt((screen_x_mars - CENTER[0])**2 + (screen_y_mars - CENTER[1])**2)
        pygame.gfxdraw.circle(screen, CENTER[0], CENTER[1], int(distance_mars), (205, 50, 50))
        # Draw Mars
        pygame.draw.circle(screen, (255, 100, 100), (screen_x_mars, screen_y_mars), 5)

   
      

        # Display orbit type text
       



    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(60)  # Limit to 60 FPS

pygame.quit()
