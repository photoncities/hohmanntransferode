import pygame
import numpy as np
import pygame.gfxdraw
from scipy.integrate import ode
import datetime

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
R_SOI_EARTH = AU * (5.972e24 / 1.989e30)**(2/5)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True
frame_count = 0

fast_dt = 3600 * 6
slow_dt = 360
dt_vis = fast_dt
speed_toggle = False

hohmann_burn_active = False
hohmann_burn_complete = False
cumulative_dv = 0.0
target_dv = 0.0
substeps = 6  # Number of sub-steps in each frame (10 minutes per sub-step if dt_vis is 3600)
timestamp = None
hohmann_ready = False

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
orbital_radius_SAT = R_earth + 400000e+2  
sat_x = earth_x + orbital_radius_SAT
sat_y = earth_y
sat_v = np.sqrt(G * earth.mass / orbital_radius_SAT)
sat_vx = earth_vx
sat_vy = earth_vy + sat_v
trail = []  # This stores positions after the burn is complete
recent_distances = []
max_history = 50  # tweak based on frame rate
apogee_ready = False
first = True
cumulative_circ_dv = 0.0
circularization_burn_complete = False
final_tweak_complete = False

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

            # Skip Earth's gravity if Satellite is outside its SOI
            if not (bodies[i].name == "Satellite" and ((bodies[j].name == "Earth" and r > R_SOI_EARTH) or (bodies[j].name == "Mars" and r > R_SOI_EARTH))):
                axi += G * mj * dx / r**3
                ayi += G * mj * dy / r**3
      

        if bodies[i].name == "Satellite":
            axi += satellite.thrustX / satellite.mass
            ayi += satellite.thrustY / satellite.mass
        derivatives.extend([vxi, vyi, axi, ayi])
    return derivatives

solver = ode(lambda t, y: equations(t, y, masses)).set_integrator(
    'dop853', nsteps=10000, atol=8, rtol=1e-6
)

solver.set_initial_value(flatten_states(bodies), 0)

def calculate_hohmann_phase_angle(r1, r2):
    a = (r1 + r2) / 2
    phase_angle_rad = np.pi * (1 - np.sqrt((r1 + r2)**3 / (8 * r2**3)))
    return phase_angle_rad  # radians

# Calculate phase angle for Earth-to-Mars transfer


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
    # if dist > 0:
    #     dir_unit = np.array([dx, dy]) / dist
    #     satellite.thrustX = 100 * dir_unit[0]
    #     satellite.thrustY = 100 * dir_unit[1]
    # satellite.thrustX = 10

    screen.fill((0, 0, 0))
    font = pygame.font.SysFont(None, 24)
    screen.blit(font.render(f"Time: {solver.t:.2f} s", True, (255, 255, 255)), (10, 10))
    radius_km = orbital_radius_SAT / 1e3
    text_surface = font.render(f"Orbital Radius: {radius_km:.2f} km", True, (255, 255, 255))
    screen.blit(text_surface, (10, 30))

       
       # Compute angle from Sun to Earth and Sun to Mars
    sun = bodies[0]
    mars = bodies[2]

    sx, sy = sun.get_position()
    ex, ey = earth.get_position()
    mx, my = mars.get_position()

    earth_angle = np.arctan2(ey, ex)
    mars_angle = np.arctan2(my, mx)

    # Normalize angles to 0–2π
    earth_angle %= 2 * np.pi
    mars_angle %= 2 * np.pi

    r_earth = np.sqrt((earth_x - sx)**2 + (earth_y - sy)**2)
    r_mars = np.sqrt((mx - sx)**2 + (my - sy)**2)
    # Compute angle Mars is ahead of Earth (circular prograde assumption)
    angle_diff = (mars_angle - earth_angle) % (2 * np.pi)

    # Target Hohmann angle in radians
    hohmann_angle = calculate_hohmann_phase_angle(r_earth,  r_mars)

        # Vectors from Earth to Satellite and Earth to Sun
    r_es = np.array([sat_x - earth_x, sat_y - earth_y])
    r_ec = np.array([sx - earth_x, sy - earth_y])  # Earth to Sun

    # Normalize both
    r_es_norm = r_es / np.linalg.norm(r_es)
    r_ec_norm = r_ec / np.linalg.norm(r_ec)

    # Dot product close to -1 → apogee condition
    alignment = np.dot(r_es_norm, r_ec_norm)
    apogee_ready = alignment < -.7995  # adjust threshold if needed
    apogee_ready_text = font.render(f"Apogee Ready: {apogee_ready}", True, (255, 255, 255))
    screen.blit(apogee_ready_text, (10, 70))

    
    # Check if we are close enough
    if abs(angle_diff - hohmann_angle) < np.radians(2) and not hohmann_burn_complete:  # within ~2 degrees
        window_text = font.render(f"Optimal Hohmann transfer window! In: {np.degrees((angle_diff - hohmann_angle)% (2*np.pi)):.2f}°", True, (0, 255, 0))
        dt_vis = slow_dt
        screen.blit(window_text, (10, 80))
    elif not hohmann_burn_complete:
        waiting_text = font.render(f"Hohmann window in: {np.degrees((angle_diff - hohmann_angle)% (2*np.pi)):.2f}°", True, (255, 255, 255))
        screen.blit(waiting_text, (10, 100))

    r1 = AU
    r2 = 1.524 * AU
    mu = G * M_sun
    

    earth = bodies[1]

    if not hohmann_burn_complete:
        
                # Trigger burn slightly *before* ideal angle
        early_margin = np.radians(0.05)  # start burn ~0.05° early
        if hohmann_angle <= angle_diff < hohmann_angle + early_margin:
            hohmann_ready = True

        if hohmann_ready and apogee_ready:


            if not hohmann_burn_active:
                # Earth's gravitational parameter
                mu_earth = G * earth.mass

                # Satellite current orbital radius around Earth
                r_sat_earth = np.sqrt((sat_x - earth_x)**2 + (sat_y - earth_y)**2)

                # Orbital velocities around Sun
                v_earth_sun = np.sqrt(G * M_sun / AU)
                a_transfer = (AU + 1.524 * AU) / 2
                v_transfer_sun = np.sqrt(G * M_sun * (2 / AU - 1 / a_transfer))
                v_inf = v_transfer_sun - v_earth_sun  # should be ~2.9 km/s


                # Satellite current circular orbital velocity around Earth
                v_circular_sat = np.sqrt(mu_earth / r_sat_earth)

                # Earth's escape velocity at satellite altitude
                v_escape_sat = np.sqrt(2 * mu_earth / r_sat_earth)

                # Total velocity required at satellite altitude
                v_total_needed = np.sqrt(v_escape_sat**2 + v_inf**2)

                # Target Δv for satellite
                target_dv = v_total_needed - v_circular_sat

                cumulative_dv = 0.0
                hohmann_burn_active = True

    if hohmann_burn_active and not hohmann_burn_complete:
        dt_vis = slow_dt / 100  # Precision timestep for burn

        
        # Burn in prograde direction relative to Earth (Earth-centered frame)
        rel_vx = satellite.state[2] - earth.state[2]
        rel_vy = satellite.state[3] - earth.state[3]
        rel_speed = np.hypot(rel_vx, rel_vy)

        tx = rel_vx / rel_speed
        ty = rel_vy / rel_speed


        thrust_mag = 5000  # N
        satellite.thrustX = tx * thrust_mag
        satellite.thrustY = ty * thrust_mag

        dv_gain = thrust_mag * dt_vis / satellite.mass
        cumulative_dv += dv_gain

        if cumulative_dv >= target_dv:
            satellite.thrustX = 0
            satellite.thrustY = 0
            hohmann_burn_complete = True
            hohmann_burn_active = False
            dt_vis = 360 * 3

   
    sat_x, sat_y = satellite.get_position()
    mars_x, mars_y = mars.get_position()
    distance_to_mars = np.sqrt((sat_x - mars_x)**2 + (sat_y - mars_y)**2)

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp is None else timestamp# Get the current real-world timestamp
    filename = f"distance_to_mars_{timestamp}.txt"
    recorded_distance = 0

    

                
    # Display burn progress
    if hohmann_burn_active:
        burn_text = font.render(f"Burning: Δv {cumulative_dv:.2f}/{target_dv:.2f} m/s", True, (255, 255, 0))
        screen.blit(burn_text, (10, 170))
        
    elif hohmann_burn_complete:
        done_text = font.render(" Transfer burn complete", True, (0, 255, 0))
        screen.blit(done_text, (10, 170))
        # Write the distance of the satellite to Mars to a timestamped file
        if first:
            with open(filename, "w") as file:
                file.write(f"Distance to Mars: {distance_to_mars:.2f} meters\n")
                first = False
        else:
            with open(filename, "r+") as file:
                content = file.read()
                if content:
                    recorded_distance = float(content.split(":")[1].strip().split()[0])
                    if distance_to_mars < recorded_distance:
                        file.seek(0)
                        file.truncate()
                        file.write(f"Distance to Mars: {distance_to_mars:.2f} meters\n")
                else:
                    file.seek(0)
                    file.write(f"Distance to Mars: {distance_to_mars:.2f} meters\n")
            
                
           
    if hohmann_burn_complete and distance_to_mars < 2e+9 and not circularization_burn_complete:
    # sat_x, sat_y = satellite.get_position()
        mars_x, mars_y = mars.get_position()
        distance_to_mars = np.sqrt((sat_x - mars_x)**2 + (sat_y - mars_y)**2)
        mu_mars = G * mars.mass

        sat_vx, sat_vy = satellite.get_velocity()
        mars_vx, mars_vy = mars.get_velocity()

        rel_vx = sat_vx - mars_vx
        rel_vy = sat_vy - mars_vy
        rel_speed = np.hypot(rel_vx, rel_vy)
        v_circular = np.sqrt(mu_mars / distance_to_mars)
        dv_needed = rel_speed - v_circular

        if abs(dv_needed) < 0.5:
            satellite.thrustX = 0
            satellite.thrustY = 0
            circularization_burn_complete = True

            
          
        else:
            tx = -rel_vx / rel_speed
            ty = -rel_vy / rel_speed
            thrust_mag = np.clip(dv_needed * satellite.mass / dt_vis, 0, 3000)
            satellite.thrustX = tx * thrust_mag
            satellite.thrustY = ty * thrust_mag

    # if circularization_burn_complete and not final_tweak_complete:
    #     sat_x, sat_y = satellite.get_position()
    #     mars_x, mars_y = mars.get_position()
    #     distance_to_mars = np.sqrt((sat_x - mars_x)**2 + (sat_y - mars_y)**2)
    #     mu_mars = G * mars.mass

    #     sat_vx, sat_vy = satellite.get_velocity()
    #     mars_vx, mars_vy = mars.get_velocity()

    #     rel_vx = sat_vx - mars_vx
    #     rel_vy = sat_vy - mars_vy
    #     rel_speed = np.hypot(rel_vx, rel_vy)
    #     v_circular = np.sqrt(mu_mars / distance_to_mars)
    #     dv_adjust = rel_speed - v_circular

    #     if abs(dv_adjust) < 0.2:
    #         satellite.thrustX = 0
    #         satellite.thrustY = 0
    #         final_tweak_complete = True
    #     else:
    #         tx = -rel_vx / rel_speed if dv_adjust > 0 else rel_vx / rel_speed
    #         ty = -rel_vy / rel_speed if dv_adjust > 0 else rel_vy / rel_speed
    #         thrust_mag = np.clip(abs(dv_adjust) * satellite.mass / dt_vis, 0, 1000)
    #         satellite.thrustX = tx * thrust_mag
    #         satellite.thrustY = ty * thrust_mag



   

    if solver.successful():
        solver.integrate(solver.t + dt_vis)
        new_state = solver.y
        for i, b in enumerate(bodies):
            b.update_state(new_state[i * 4:(i + 1) * 4])

        # Collision check: satellite inside Earth
        sat_x, sat_y = satellite.get_position()
        earth_x, earth_y = earth.get_position()
        distance = np.sqrt((sat_x - earth_x)**2 + (sat_y - earth_y)**2)

        if distance <= R_earth:
            crash_text = font.render("Simulation Failure; collision detected.", True, (255, 50, 50))
            screen.blit(crash_text, (10, 50))
            running = False

           

        # if frame_count % 5 == 0:
        #     pred_trail = predict_trajectory(satellite, 5, 3600 * 4)

        for b in bodies:
            x, y = b.get_position()
            sx = int(CENTER[0] + x * SCALE)
            sy = int(CENTER[1] - y * SCALE)
            color = (255, 255, 0) if b.name == "Sun" else (0, 0, 255) if b.name == "Earth" else (255, 100, 100) if b.name == "Mars" else (0, 255, 0)
            pygame.draw.circle(screen, color, (sx, sy), 5 if b.name != "Sun" else 10)
            if b.name == "Satellite":
                tx, ty = -satellite.thrustX, -satellite.thrustY
                mag = np.sqrt(tx**2 + ty**2)
                if mag > 0:
                    tx /= mag
                    ty /= mag
                    tip = (sx + tx * 12, sy - ty * 12)
                    left = (sx - ty * 5, sy - tx * 5)
                    right = (sx + ty * 5, sy + tx * 5)
                    pygame.draw.polygon(screen, (0, 255, 255), [tip, left, right])
                # Sun position on screen
        sun_screen = (int(CENTER[0]), int(CENTER[1]))

        # Earth screen position
        earth_x, earth_y = earth.get_position()
        earth_screen = (int(CENTER[0] + earth_x * SCALE), int(CENTER[1] - earth_y * SCALE))

        # Mars screen position
        mars_x, mars_y = mars.get_position()
        mars_screen = (int(CENTER[0] + mars_x * SCALE), int(CENTER[1] - mars_y * SCALE))

        # Draw lines from Sun to Earth and Mars
        pygame.draw.line(screen, (100, 100, 255), sun_screen, earth_screen, 2)
        pygame.draw.line(screen, (255, 100, 100), sun_screen, mars_screen, 2)

        # Optional: draw angle arc at Sun (approximate)
        def draw_angle_arc(surface, center, r, start_angle, end_angle, color, steps=50):
            for i in range(steps):
                t1 = start_angle + (end_angle - start_angle) * i / steps
                t2 = start_angle + (end_angle - start_angle) * (i + 1) / steps
                x1 = center[0] + r * np.cos(t1)
                y1 = center[1] - r * np.sin(t1)
                x2 = center[0] + r * np.cos(t2)
                y2 = center[1] - r * np.sin(t2)
                pygame.draw.line(surface, color, (x1, y1), (x2, y2), 1)


        # arc_start = mars_angle
        # remaining_angle = (angle_diff - hohmann_angle) % (2 * np.pi)
        # arc_end = mars_angle + remaining_angle

        # if remaining_angle > np.pi:
        #     arc_start, arc_end = arc_end, arc_start + 2 * np.pi  # draw shortest arc

        # draw_angle_arc(screen, sun_screen, 80, arc_start, arc_end, (255, 255, 0))
        # Draw arc for angle between Earth and Mars

        # Draw hollow circles for orbital radii of planets centered on the Sun
        for b in bodies:
            if b.name != "Sun" and b.name != "Satellite":
                orbital_radius_SAT = np.sqrt((b.state[0] - sx)**2 + (b.state[1] - sy)**2)
                pygame.gfxdraw.aacircle(
                    screen,
                    int(CENTER[0]),
                    int(CENTER[1]),
                    int(orbital_radius_SAT * SCALE),
                    (100, 100, 100)
                )

        draw_angle_arc(screen, sun_screen, 60, earth_angle, mars_angle, (0, 255, 0))

        # for pt in pred_trail:
        #     pygame.draw.circle(screen, (200, 200, 200), pt[:2], 1)

        # Camera rendering
        # Camera rendering
        cam_surface = pygame.Surface((CAM_WIDTH, CAM_HEIGHT))
        cam_surface.fill((20, 20, 20))

        # Calculate satellite center on cam
        sat_cx = CAM_WIDTH // 2
        sat_cy = CAM_HEIGHT // 2

        # Draw all bodies relative to satellite
        for b in bodies:
            bx, by = b.get_position()
            dx = (bx - sat_x) * CAM_SCALE
            dy = (by - sat_y) * CAM_SCALE
            cx = sat_cx + dx
            cy = sat_cy - dy

            color = (255, 255, 0) if b.name == "Sun" else (0, 0, 255) if b.name == "Earth" else (255, 100, 100) if b.name == "Mars" else (0, 255, 0)
            pygame.draw.circle(cam_surface, color, (int(cx), int(cy)), 4 if b.name != "Sun" else 6)

            if b.name == "Satellite":
                # Draw thrust triangle offset from satellite center
                tx, ty = -satellite.thrustX, satellite.thrustY
                mag = np.sqrt(tx**2 + ty**2)
                if mag > 0:
                    tx /= mag
                    ty /= mag
                    offset = 10
                    length = 7
                    width = 3.5

                    ox = cx + tx * offset
                    oy = cy + ty * offset

                    tip = (ox + tx * length, oy + ty * length)
                    left = (ox - ty * width, oy + tx * width)
                    right = (ox + ty * width, oy - tx * width)

                    pygame.draw.polygon(cam_surface, (0, 255, 255), [tip, left, right])

        # Draw prediction trail
        # for pt in pred_trail:
        #     px = (pt[2] - sat_x) * CAM_SCALE + CAM_WIDTH // 2
        #     py = (pt[3] - sat_y) * CAM_SCALE + CAM_HEIGHT // 2
        #     pygame.draw.circle(cam_surface, (0, 200, 0), (int(px), int(py)), 2)

        # Draw cam border + center mark
        pygame.draw.rect(screen, (255, 255, 255), (*CAM_POS, CAM_WIDTH, CAM_HEIGHT), 1)
        screen.blit(cam_surface, CAM_POS)
        # pygame.draw.circle(screen, (255, 0, 0), (CAM_POS[0] + CAM_WIDTH // 2, CAM_POS[1] + CAM_HEIGHT // 2), 2)

        if hohmann_burn_complete:
            # Track the satellite’s position after the burn
            sat_x, sat_y = satellite.get_position()
            # trail.append((sat_x, sat_y))
        for pos in trail:
            screen_x = int(CENTER[0] + pos[0] * SCALE)
            screen_y = int(CENTER[1] - pos[1] * SCALE)
            pygame.draw.circle(screen, (200, 200, 200), (screen_x, screen_y), 2)

        # Limit the trail length to avoid performance issues
        # if len(trail) > 1000:  # Adjust the limit as needed
        #     trail = trail[-1000:]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
            speed_toggle = not speed_toggle
            dt_vis = slow_dt if speed_toggle else fast_dt

        elif event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if (CAM_POS[0] <= mx <= CAM_POS[0] + CAM_WIDTH and
                CAM_POS[1] <= my <= CAM_POS[1] + CAM_HEIGHT):
                zoom_factor = 0.9 if event.y < 0 else 1.1
                CAM_SCALE *= zoom_factor
                CAM_SCALE = max(1e-20, min(CAM_SCALE, 1e-2))  # clamp to avoid insanity


    pygame.display.flip()
    clock.tick(60)
    frame_count += 1

while not running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            running = True
