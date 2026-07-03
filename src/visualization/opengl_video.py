"""OpenGL-based video exporter for the surface simulation.

Renders the same visualization as visualization_opengl.py but saves to MP4.

Requires: pip install PyOpenGL PyOpenGL_accelerate glfw imageio imageio-ffmpeg
"""

import numpy as np
import glfw
from OpenGL.GL import *
import ctypes
import imageio

from src.physics import catenary as cat
from src.visualization import compute_cell_vertex_colors


# Vertex shader - transforms vertices and passes data to fragment shader.
# Attribute 2 (aColor) is an optional per-vertex tint. When not set per-vertex,
# the generic attribute value (1, 1, 1) is used and acts as a no-op multiplier
# so the existing single-uniform-colour rendering is preserved.
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

out vec3 FragPos;
out vec3 Normal;
out vec3 VertColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    VertColor = aColor;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

# Fragment shader - computes lighting and color
FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec3 VertColor;

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

    vec3 result = (ambient + diffuse + specular) * objectColor * VertColor;
    FragColor = vec4(result, 1.0);
}
"""


def create_shader_program():
    """Compile and link the shader program."""
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, VERTEX_SHADER)
    glCompileShader(vertex_shader)
    if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(vertex_shader).decode())

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, FRAGMENT_SHADER)
    glCompileShader(fragment_shader)
    if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(fragment_shader).decode())

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(shader_program).decode())

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program


def create_cylinder_mesh(sectors=16):
    """Generate a unit cylinder mesh (radius 1, base at z=0, top at z=1) with normals.

    Includes side wall and top cap; bottom cap is omitted since rods sit on z=0.
    """
    vertices = []
    normals = []
    indices = []

    # Side wall: pairs of (bottom, top) ring vertices with radial normals
    for j in range(sectors + 1):
        angle = j * 2 * np.pi / sectors
        cx, cy = np.cos(angle), np.sin(angle)
        vertices.extend([cx, cy, 0.0])
        normals.extend([cx, cy, 0.0])
        vertices.extend([cx, cy, 1.0])
        normals.extend([cx, cy, 0.0])

    for j in range(sectors):
        b0 = 2 * j
        t0 = 2 * j + 1
        b1 = 2 * (j + 1)
        t1 = 2 * (j + 1) + 1
        indices.extend([b0, b1, t0])
        indices.extend([t0, b1, t1])

    # Top cap: ring + center, normals pointing +Z
    base = len(vertices) // 3
    for j in range(sectors):
        angle = j * 2 * np.pi / sectors
        vertices.extend([np.cos(angle), np.sin(angle), 1.0])
        normals.extend([0.0, 0.0, 1.0])
    center = len(vertices) // 3
    vertices.extend([0.0, 0.0, 1.0])
    normals.extend([0.0, 0.0, 1.0])

    for j in range(sectors):
        indices.extend([base + j, center, base + (j + 1) % sectors])

    return (np.array(vertices, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(indices, dtype=np.uint32))


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

    vertices = []
    normals = []
    indices = []

    for i in range(nx):
        for j in range(ny):
            vertices.extend([X[i], Y[j], Z[i, j]])

    for i in range(nx):
        for j in range(ny):
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

            normal = np.array([-dzdx, -dzdy, 1.0])
            normal = normal / np.linalg.norm(normal)
            normals.extend(normal.tolist())

    for i in range(nx - 1):
        for j in range(ny - 1):
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


class OpenGLVideoExporter:
    """Exports simulation to MP4 video using OpenGL offscreen rendering."""

    def __init__(self, rodsstates, ballsstates, ballradiuses, config,
                 resolution=10, width=1200, height=800, fps=30,
                 rod_radius=0.02, channels=None):
        from src.visualization import colors_from_radii
        self.rodsstates = rodsstates
        self.ballsstates = ballsstates
        self.ballradiuses = ballradiuses
        self.ball_colors = colors_from_radii(ballradiuses)
        self.config = config
        self.resolution = resolution
        self.width = width
        self.height = height
        self.fps = fps
        self.rod_radius = rod_radius
        # Optional per-cell scalar channels for surface tinting. None → existing visuals.
        self.channels = channels
        self._weight_frames = channels.get("weight") if channels else None

        # Camera parameters (same as interactive visualizer)
        grid_center_x = (config.GRIDSIZEX - 1) * config.D_RODS / 2
        grid_center_y = (config.GRIDSIZEY - 1) * config.D_RODS / 2
        self.camera_distance = max(config.GRIDSIZEX, config.GRIDSIZEY) * config.D_RODS * 0.7
        self.camera_angle_h = -np.pi / 2  # horizontal angle (90 degrees left)
        self.camera_angle_v = np.radians(10)  # vertical angle (10 degrees from top)
        self.camera_target = np.array([grid_center_x, grid_center_y, 0.0], dtype=np.float32)

    def init_gl(self):
        """Initialize OpenGL context with hidden window for offscreen rendering."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # Hidden window for offscreen rendering

        self.window = glfw.create_window(self.width, self.height,
                                         "Offscreen", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # Create framebuffer for offscreen rendering
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # Color texture
        self.color_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_texture, 0)

        # Depth renderbuffer
        self.depth_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_rbo)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer is not complete")

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)

        # Compile shaders
        self.shader_program = create_shader_program()

        # Create sphere mesh for balls
        sphere_verts, sphere_normals, sphere_indices = create_sphere_mesh(1.0, 16, 16)
        self.sphere_index_count = len(sphere_indices)

        # Create sphere VAO
        self.sphere_vao = glGenVertexArrays(1)
        glBindVertexArray(self.sphere_vao)

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

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        # Create cylinder mesh for rods
        cyl_verts, cyl_normals, cyl_indices = create_cylinder_mesh(16)
        self.cylinder_index_count = len(cyl_indices)

        self.cylinder_vao = glGenVertexArrays(1)
        glBindVertexArray(self.cylinder_vao)

        cyl_data = np.zeros(len(cyl_verts) + len(cyl_normals), dtype=np.float32)
        cyl_data[0::6] = cyl_verts[0::3]
        cyl_data[1::6] = cyl_verts[1::3]
        cyl_data[2::6] = cyl_verts[2::3]
        cyl_data[3::6] = cyl_normals[0::3]
        cyl_data[4::6] = cyl_normals[1::3]
        cyl_data[5::6] = cyl_normals[2::3]

        cyl_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, cyl_vbo)
        glBufferData(GL_ARRAY_BUFFER, cyl_data.nbytes, cyl_data, GL_STATIC_DRAW)

        cyl_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cyl_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, cyl_indices.nbytes, cyl_indices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        # Create surface VAO
        self.surface_vao = glGenVertexArrays(1)
        self.surface_vbo = glGenBuffers(1)
        self.surface_color_vbo = glGenBuffers(1)
        self.surface_ebo = glGenBuffers(1)

        # Default the generic vertex-attribute value for the optional per-vertex
        # color to white so geometry without a color buffer (sphere, cylinder)
        # renders untinted.
        glVertexAttrib3f(2, 1.0, 1.0, 1.0)

    def update_surface_mesh(self, frame):
        """Update the surface mesh for the given frame."""
        rods = self.rodsstates[frame]
        verts, normals, indices = compute_surface_mesh(rods, self.config, self.resolution)
        self.surface_index_count = len(indices)

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

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        # Per-vertex color attribute, refreshed each frame from the optional
        # weight channel. When no channels are present this returns all-white,
        # which combined with the existing objectColor uniform yields the
        # previous appearance unchanged.
        nx = (self.config.GRIDSIZEX - 1) * self.resolution + 1
        ny = (self.config.GRIDSIZEY - 1) * self.resolution + 1
        weight_frame = (self._weight_frames[frame]
                        if self._weight_frames is not None
                        and frame < len(self._weight_frames)
                        else None)
        target = float(getattr(self.config, "TARGET_WEIGHT", 0.0))
        cell_colors = compute_cell_vertex_colors(
            weight_frame,
            self.config.GRIDSIZEX - 1, self.config.GRIDSIZEY - 1,
            nx, ny, self.resolution, target,
        )
        glBindBuffer(GL_ARRAY_BUFFER, self.surface_color_vbo)
        glBufferData(GL_ARRAY_BUFFER, cell_colors.nbytes, cell_colors, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.surface_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)

    def get_camera_position(self):
        """Calculate camera position from angles and distance."""
        x = self.camera_target[0] + self.camera_distance * np.sin(self.camera_angle_v) * np.cos(self.camera_angle_h)
        y = self.camera_target[1] + self.camera_distance * np.sin(self.camera_angle_v) * np.sin(self.camera_angle_h)
        z = self.camera_target[2] + self.camera_distance * np.cos(self.camera_angle_v)
        return np.array([x, y, z], dtype=np.float32)

    def render_frame(self, frame):
        """Render a single frame and return as numpy array."""
        self.update_surface_mesh(frame)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.shader_program)

        aspect = self.width / self.height
        projection = perspective_matrix(np.radians(60), aspect, 0.1, 100.0)

        camera_pos = self.get_camera_position()
        view = look_at_matrix(camera_pos, self.camera_target, np.array([0, 0, 1], dtype=np.float32))

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

        light_pos = self.camera_target + np.array([0, 0, 10], dtype=np.float32)
        glUniform3fv(light_loc, 1, light_pos)
        glUniform3fv(view_pos_loc, 1, camera_pos)

        # Draw surface (matte)
        model = np.eye(4, dtype=np.float32)
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
        glUniform3f(color_loc, 0.85, 0.85, 0.85)
        glUniform1f(spec_strength_loc, 0.3)
        glUniform1f(shininess_loc, 16.0)

        glBindVertexArray(self.surface_vao)
        glDrawElements(GL_TRIANGLES, self.surface_index_count, GL_UNSIGNED_INT, None)

        # Draw rods (light gray cylinders from z=0 up to rod height)
        glUniform3f(color_loc, 0.75, 0.75, 0.78)
        glUniform1f(spec_strength_loc, 0.4)
        glUniform1f(shininess_loc, 32.0)
        rods = self.rodsstates[frame]
        glBindVertexArray(self.cylinder_vao)
        for i in range(self.config.GRIDSIZEX):
            for j in range(self.config.GRIDSIZEY):
                rx, ry, rz = rods[i, j]
                # Stop the cylinder slightly below the fabric attachment so the
                # finite-radius cylinder wall doesn't protrude through the sag.
                visible_height = max(rz - 2.0 * self.rod_radius, 0.0)
                model = (translation_matrix(rx, ry, 0.0)
                         @ scale_matrix(self.rod_radius, self.rod_radius, visible_height))
                glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
                glDrawElements(GL_TRIANGLES, self.cylinder_index_count, GL_UNSIGNED_INT, None)

        # Draw balls (shiny), colored by weight class
        glUniform1f(spec_strength_loc, 1.0)
        glUniform1f(shininess_loc, 128.0)
        balls_pos = self.ballsstates[frame]

        for i, pos in enumerate(balls_pos):
            radius = self.ballradiuses[i]
            color = self.ball_colors[i]
            glUniform3f(color_loc, float(color[0]), float(color[1]), float(color[2]))
            model = translation_matrix(pos[0], pos[1], pos[2]) @ scale_matrix(radius, radius, radius)
            glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)

            glBindVertexArray(self.sphere_vao)
            glDrawElements(GL_TRIANGLES, self.sphere_index_count, GL_UNSIGNED_INT, None)

        # Read pixels from framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        # Flip vertically (OpenGL has origin at bottom-left)
        image = np.flipud(image)

        return image

    def export(self, output_path):
        """Export simulation to MP4 video."""
        self.init_gl()

        total_frames = len(self.rodsstates)
        print(f"Exporting {total_frames} frames to {output_path}...")

        writer = imageio.get_writer(output_path, fps=self.fps, codec='libx264',
                                    pixelformat='yuv420p', quality=8)

        for frame in range(total_frames):
            if frame % 50 == 0:
                print(f"  Frame {frame}/{total_frames}")

            image = self.render_frame(frame)
            writer.append_data(image)

        writer.close()
        glfw.terminate()

        print(f"Video saved to {output_path}")


def export_video(rodsstates, ballsstates, ballradiuses, config,
                 output_path="output/simulation.mp4",
                 resolution=10, width=1200, height=800, fps=30,
                 channels=None):
    """Export simulation to MP4 video.

    Args:
        rodsstates: List of rod arrays, one per timestep.
        ballsstates: List of ball position arrays (N, 3), one per timestep.
        ballradiuses: Array of ball radii.
        config: SimConfig instance.
        output_path: Path to save the MP4 file.
        resolution: Surface grid points per module edge.
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Frames per second.
        channels: Optional dict of per-cell scalar channels. Currently the
            "weight" channel tints the surface. When None, the surface renders
            with the original single-colour appearance.
    """
    exporter = OpenGLVideoExporter(
        rodsstates, ballsstates, ballradiuses, config,
        resolution=resolution, width=width, height=height, fps=fps,
        channels=channels,
    )
    exporter.export(output_path)
