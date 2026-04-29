"""OpenGL-based 3D visualization of the surface simulation.

Requires: pip install PyOpenGL PyOpenGL_accelerate glfw
"""

import numpy as np
import glfw
from OpenGL.GL import *
import ctypes

from src.physics import catenary as cat


# Vertex shader - transforms vertices and passes data to fragment shader
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

# Fragment shader - computes lighting and color
FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform float specularStrength;
uniform float shininess;

void main()
{
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);

    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
"""


def create_shader_program():
    """Compile and link the shader program."""
    # Compile vertex shader
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, VERTEX_SHADER)
    glCompileShader(vertex_shader)
    if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(vertex_shader).decode())

    # Compile fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, FRAGMENT_SHADER)
    glCompileShader(fragment_shader)
    if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(fragment_shader).decode())

    # Link program
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(shader_program).decode())

    # Clean up shaders (they're linked into the program now)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program


def create_sphere_mesh(radius=1.0, sectors=16, stacks=16):
    """Generate a UV sphere mesh with normals."""
    vertices = []
    normals = []
    indices = []

    for i in range(stacks + 1):
        stack_angle = np.pi / 2 - i * np.pi / stacks
        xy = radius * np.cos(stack_angle)
        z = radius * np.sin(stack_angle)

        for j in range(sectors + 1):
            sector_angle = j * 2 * np.pi / sectors
            x = xy * np.cos(sector_angle)
            y = xy * np.sin(sector_angle)

            vertices.extend([x, y, z])
            # Normal is just the normalized position for a unit sphere
            length = np.sqrt(x*x + y*y + z*z)
            if length > 0:
                normals.extend([x/length, y/length, z/length])
            else:
                normals.extend([0, 0, 1])

    for i in range(stacks):
        k1 = i * (sectors + 1)
        k2 = k1 + sectors + 1

        for j in range(sectors):
            if i != 0:
                indices.extend([k1, k2, k1 + 1])
            if i != stacks - 1:
                indices.extend([k1 + 1, k2, k2 + 1])
            k1 += 1
            k2 += 1

    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(indices, dtype=np.uint32)


def compute_surface_mesh(rods, config, resolution=10):
    """Compute surface vertices and normals for OpenGL rendering."""
    nx = (config.GRIDSIZEX - 1) * resolution + 1
    ny = (config.GRIDSIZEY - 1) * resolution + 1

    X = np.linspace(0, (config.GRIDSIZEX - 1) * config.D_RODS, nx)
    Y = np.linspace(0, (config.GRIDSIZEY - 1) * config.D_RODS, ny)
    Z = np.zeros((nx, ny), dtype=np.float32)

    # Compute z heights
    for i in range(config.GRIDSIZEX - 1):
        for j in range(config.GRIDSIZEY - 1):
            x0 = i * config.D_RODS
            y0 = j * config.D_RODS

            rod_sw = rods[i, j, 2]
            rod_se = rods[i + 1, j, 2]
            rod_nw = rods[i, j + 1, 2]
            rod_ne = rods[i + 1, j + 1, 2]

            cat_w = cat.findcatenaryparameters(config.D_FABRIC, config.D_RODS, rod_sw, rod_nw)
            cat_e = cat.findcatenaryparameters(config.D_FABRIC, config.D_RODS, rod_se, rod_ne)

            ix_start = i * resolution
            ix_end = (i + 1) * resolution + 1
            iy_start = j * resolution
            iy_end = (j + 1) * resolution + 1

            for ii in range(ix_start, ix_end):
                local_x = X[ii] - x0
                for jj in range(iy_start, iy_end):
                    local_y = Y[jj] - y0
                    h_w = cat.catenary(local_y, cat_w)
                    h_e = cat.catenary(local_y, cat_e)
                    cat_we = cat.findcatenaryparameters(config.D_FABRIC, config.D_RODS, h_w, h_e)
                    Z[ii, jj] = cat.catenary(local_x, cat_we)

    # Generate vertices, normals, and indices for triangle mesh
    vertices = []
    normals = []
    indices = []

    # Create vertex grid
    for i in range(nx):
        for j in range(ny):
            vertices.extend([X[i], Y[j], Z[i, j]])

    # Compute normals using finite differences
    for i in range(nx):
        for j in range(ny):
            # Get neighboring z values for gradient
            if i > 0 and i < nx - 1:
                dzdx = (Z[i+1, j] - Z[i-1, j]) / (2 * (X[1] - X[0]))
            elif i == 0:
                dzdx = (Z[i+1, j] - Z[i, j]) / (X[1] - X[0])
            else:
                dzdx = (Z[i, j] - Z[i-1, j]) / (X[1] - X[0])

            if j > 0 and j < ny - 1:
                dzdy = (Z[i, j+1] - Z[i, j-1]) / (2 * (Y[1] - Y[0]))
            elif j == 0:
                dzdy = (Z[i, j+1] - Z[i, j]) / (Y[1] - Y[0])
            else:
                dzdy = (Z[i, j] - Z[i, j-1]) / (Y[1] - Y[0])

            # Normal = cross product of tangent vectors
            # tangent_x = (1, 0, dzdx), tangent_y = (0, 1, dzdy)
            # normal = tangent_x x tangent_y = (-dzdx, -dzdy, 1)
            normal = np.array([-dzdx, -dzdy, 1.0])
            normal = normal / np.linalg.norm(normal)
            normals.extend(normal.tolist())

    # Generate triangle indices
    for i in range(nx - 1):
        for j in range(ny - 1):
            # Two triangles per quad
            v0 = i * ny + j
            v1 = (i + 1) * ny + j
            v2 = (i + 1) * ny + (j + 1)
            v3 = i * ny + (j + 1)

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    return (np.array(vertices, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(indices, dtype=np.uint32))


def perspective_matrix(fov, aspect, near, far):
    """Create a perspective projection matrix."""
    f = 1.0 / np.tan(fov / 2)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2 * far * near) / (near - far)
    mat[3, 2] = -1
    return mat


def look_at_matrix(eye, center, up):
    """Create a view matrix using look-at parameters."""
    f = center - eye
    f = f / np.linalg.norm(f)

    s = np.cross(f, up)
    s = s / np.linalg.norm(s)

    u = np.cross(s, f)

    mat = np.eye(4, dtype=np.float32)
    mat[0, :3] = s
    mat[1, :3] = u
    mat[2, :3] = -f
    mat[0, 3] = -np.dot(s, eye)
    mat[1, 3] = -np.dot(u, eye)
    mat[2, 3] = np.dot(f, eye)
    return mat


def translation_matrix(tx, ty, tz):
    """Create a translation matrix."""
    mat = np.eye(4, dtype=np.float32)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat


def scale_matrix(sx, sy, sz):
    """Create a scale matrix."""
    mat = np.eye(4, dtype=np.float32)
    mat[0, 0] = sx
    mat[1, 1] = sy
    mat[2, 2] = sz
    return mat


class OpenGLVisualizer:
    """Interactive OpenGL visualizer for the surface simulation."""

    def __init__(self, rodsstates, ballsstates, ballradiuses, config,
                 resolution=10, width=1200, height=800):
        self.rodsstates = rodsstates
        self.ballsstates = ballsstates
        self.ballradiuses = ballradiuses
        self.config = config
        self.resolution = resolution
        self.width = width
        self.height = height

        self.current_frame = 0
        self.playing = True
        self.frame_delay = 0.05  # seconds between frames
        self.last_frame_time = 0

        # Camera parameters
        grid_center_x = (config.GRIDSIZEX - 1) * config.D_RODS / 2
        grid_center_y = (config.GRIDSIZEY - 1) * config.D_RODS / 2
        self.camera_distance = max(config.GRIDSIZEX, config.GRIDSIZEY) * config.D_RODS * 0.7
        self.camera_angle_h = -np.pi / 2  # horizontal angle (90 degrees left)
        self.camera_angle_v = np.radians(10)  # vertical angle (10 degrees from top)
        self.camera_target = np.array([grid_center_x, grid_center_y, 0.0], dtype=np.float32)

        # Mouse state
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_pressed = False

    def init_gl(self):
        """Initialize OpenGL context and resources."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        self.window = glfw.create_window(self.width, self.height,
                                         "Surface Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # Compile shaders
        self.shader_program = create_shader_program()

        # Create sphere mesh for balls
        sphere_verts, sphere_normals, sphere_indices = create_sphere_mesh(1.0, 16, 16)
        self.sphere_index_count = len(sphere_indices)

        # Create sphere VAO
        self.sphere_vao = glGenVertexArrays(1)
        glBindVertexArray(self.sphere_vao)

        # Interleave vertex and normal data
        sphere_data = np.zeros(len(sphere_verts) + len(sphere_normals), dtype=np.float32)
        sphere_data[0::6] = sphere_verts[0::3]
        sphere_data[1::6] = sphere_verts[1::3]
        sphere_data[2::6] = sphere_verts[2::3]
        sphere_data[3::6] = sphere_normals[0::3]
        sphere_data[4::6] = sphere_normals[1::3]
        sphere_data[5::6] = sphere_normals[2::3]

        sphere_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo)
        glBufferData(GL_ARRAY_BUFFER, sphere_data.nbytes, sphere_data, GL_STATIC_DRAW)

        sphere_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.nbytes, sphere_indices, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        # Create surface VAO (will be updated each frame)
        self.surface_vao = glGenVertexArrays(1)
        self.surface_vbo = glGenBuffers(1)
        self.surface_ebo = glGenBuffers(1)

        # Initialize surface mesh
        self.update_surface_mesh(0)

    def update_surface_mesh(self, frame):
        """Update the surface mesh for the given frame."""
        rods = self.rodsstates[frame]
        verts, normals, indices = compute_surface_mesh(rods, self.config, self.resolution)
        self.surface_index_count = len(indices)

        # Interleave vertex and normal data
        n_verts = len(verts) // 3
        surface_data = np.zeros(n_verts * 6, dtype=np.float32)
        surface_data[0::6] = verts[0::3]
        surface_data[1::6] = verts[1::3]
        surface_data[2::6] = verts[2::3]
        surface_data[3::6] = normals[0::3]
        surface_data[4::6] = normals[1::3]
        surface_data[5::6] = normals[2::3]

        glBindVertexArray(self.surface_vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.surface_vbo)
        glBufferData(GL_ARRAY_BUFFER, surface_data.nbytes, surface_data, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.surface_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

    def get_camera_position(self):
        """Calculate camera position from angles and distance."""
        x = self.camera_target[0] + self.camera_distance * np.sin(self.camera_angle_v) * np.cos(self.camera_angle_h)
        y = self.camera_target[1] + self.camera_distance * np.sin(self.camera_angle_v) * np.sin(self.camera_angle_h)
        z = self.camera_target[2] + self.camera_distance * np.cos(self.camera_angle_v)
        return np.array([x, y, z], dtype=np.float32)

    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input."""
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE:
                self.playing = not self.playing
            elif key == glfw.KEY_R:
                self.current_frame = 0
            elif key == glfw.KEY_RIGHT:
                self.current_frame = min(self.current_frame + 1, len(self.rodsstates) - 1)
            elif key == glfw.KEY_LEFT:
                self.current_frame = max(self.current_frame - 1, 0)
            elif key == glfw.KEY_UP:
                self.frame_delay = max(0.01, self.frame_delay - 0.01)
            elif key == glfw.KEY_DOWN:
                self.frame_delay = min(0.5, self.frame_delay + 0.01)

    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button input."""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pressed = (action == glfw.PRESS)
            if self.mouse_pressed:
                self.last_mouse_x, self.last_mouse_y = glfw.get_cursor_pos(window)

    def cursor_pos_callback(self, window, xpos, ypos):
        """Handle mouse movement for camera rotation."""
        if self.mouse_pressed:
            dx = xpos - self.last_mouse_x
            dy = ypos - self.last_mouse_y

            self.camera_angle_h += dx * 0.005
            self.camera_angle_v = np.clip(self.camera_angle_v + dy * 0.005, 0.1, np.pi - 0.1)

            self.last_mouse_x = xpos
            self.last_mouse_y = ypos

    def scroll_callback(self, window, xoffset, yoffset):
        """Handle scroll for zoom."""
        self.camera_distance = np.clip(
            self.camera_distance * (1 - yoffset * 0.1),
            1.0, 100.0
        )

    def framebuffer_size_callback(self, window, width, height):
        """Handle window resize."""
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)

    def render(self):
        """Render a single frame."""
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.shader_program)

        # Set up matrices
        aspect = self.width / self.height if self.height > 0 else 1.0
        projection = perspective_matrix(np.radians(60), aspect, 0.1, 100.0)

        camera_pos = self.get_camera_position()
        view = look_at_matrix(camera_pos, self.camera_target, np.array([0, 0, 1], dtype=np.float32))

        # Set uniforms
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        model_loc = glGetUniformLocation(self.shader_program, "model")
        light_loc = glGetUniformLocation(self.shader_program, "lightPos")
        view_pos_loc = glGetUniformLocation(self.shader_program, "viewPos")
        color_loc = glGetUniformLocation(self.shader_program, "objectColor")
        spec_strength_loc = glGetUniformLocation(self.shader_program, "specularStrength")
        shininess_loc = glGetUniformLocation(self.shader_program, "shininess")

        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, projection)
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)

        # Light position above the scene
        light_pos = self.camera_target + np.array([0, 0, 10], dtype=np.float32)
        glUniform3fv(light_loc, 1, light_pos)
        glUniform3fv(view_pos_loc, 1, camera_pos)

        # Draw surface (matte)
        model = np.eye(4, dtype=np.float32)
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
        glUniform3f(color_loc, 0.85, 0.85, 0.85)  # White/grey surface
        glUniform1f(spec_strength_loc, 0.3)  # Low specular for matte look
        glUniform1f(shininess_loc, 16.0)

        glBindVertexArray(self.surface_vao)
        glDrawElements(GL_TRIANGLES, self.surface_index_count, GL_UNSIGNED_INT, None)

        # Draw balls (shiny)
        glUniform3f(color_loc, 0.15, 0.15, 0.15)  # Dark grey balls
        glUniform1f(spec_strength_loc, 1.0)  # High specular for shiny look
        glUniform1f(shininess_loc, 128.0)  # High shininess for tight highlights
        balls_pos = self.ballsstates[self.current_frame]

        for i, pos in enumerate(balls_pos):
            radius = self.ballradiuses[i]
            model = translation_matrix(pos[0], pos[1], pos[2]) @ scale_matrix(radius, radius, radius)
            glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)

            glBindVertexArray(self.sphere_vao)
            glDrawElements(GL_TRIANGLES, self.sphere_index_count, GL_UNSIGNED_INT, None)

        glfw.swap_buffers(self.window)

    def run(self):
        """Main visualization loop."""
        self.init_gl()

        print("OpenGL Visualizer Controls:")
        print("  Mouse drag: Rotate camera")
        print("  Scroll: Zoom in/out")
        print("  Space: Play/Pause")
        print("  Left/Right arrows: Step frames")
        print("  Up/Down arrows: Adjust speed")
        print("  R: Reset to first frame")
        print("  Escape: Quit")

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            current_time = glfw.get_time()

            # Update animation
            if self.playing and current_time - self.last_frame_time >= self.frame_delay:
                self.current_frame = (self.current_frame + 1) % len(self.rodsstates)
                self.update_surface_mesh(self.current_frame)
                self.last_frame_time = current_time

            self.render()

            # Update window title with frame info
            title = f"Surface Simulation - Frame {self.current_frame}/{len(self.rodsstates)-1}"
            if not self.playing:
                title += " (Paused)"
            glfw.set_window_title(self.window, title)

        glfw.terminate()


def animate_simulation(rodsstates, ballsstates, ballradiuses, config,
                       resolution=10, width=1200, height=800):
    """Show an interactive OpenGL 3D visualization of the simulation.

    Args:
        rodsstates: List of rod arrays, one per timestep.
        ballsstates: List of ball position arrays (N, 3), one per timestep.
        ballradiuses: Array of ball radii.
        config: SimConfig instance.
        resolution: Surface grid points per module edge.
        width: Window width in pixels.
        height: Window height in pixels.
    """
    visualizer = OpenGLVisualizer(
        rodsstates, ballsstates, ballradiuses, config,
        resolution=resolution, width=width, height=height
    )
    visualizer.run()
