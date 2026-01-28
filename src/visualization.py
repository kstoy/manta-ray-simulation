import numpy as np
from pathlib import Path
from src.physics import catenary as cat
import pygltflib as gltf

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def generate_ball_triangles(config):
    """Generate sphere triangles using vectorized numpy operations."""
    phis = np.linspace(0, np.pi, config.TRIANGLES, endpoint=False)
    thetas = np.linspace(0, 2 * np.pi, config.TRIANGLES, endpoint=False)
    dtheta = 2 * np.pi / config.TRIANGLES
    dphi = np.pi / config.TRIANGLES

    # Create meshgrid and flatten
    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing='ij')
    phi_flat = phi_grid.ravel()
    theta_flat = theta_grid.ravel()

    # Pre-compute trig values
    sin_phi = np.sin(phi_flat)
    cos_phi = np.cos(phi_flat)
    sin_phi_dp = np.sin(phi_flat + dphi)
    cos_phi_dp = np.cos(phi_flat + dphi)

    cos_theta = np.cos(theta_flat)
    sin_theta = np.sin(theta_flat)
    cos_theta_dt = np.cos(theta_flat + dtheta)
    sin_theta_dt = np.sin(theta_flat + dtheta)

    # Each quad produces 6 vertices (2 triangles)
    n_quads = len(phi_flat)
    ball_triangles = np.zeros((n_quads * 6, 3), dtype=np.float32)

    # Vertex 0: (phi, theta)
    ball_triangles[0::6, 0] = sin_phi * cos_theta
    ball_triangles[0::6, 1] = sin_phi * sin_theta
    ball_triangles[0::6, 2] = cos_phi

    # Vertex 1: (phi, theta + dtheta)
    ball_triangles[1::6, 0] = sin_phi * cos_theta_dt
    ball_triangles[1::6, 1] = sin_phi * sin_theta_dt
    ball_triangles[1::6, 2] = cos_phi

    # Vertex 2: (phi + dphi, theta + dtheta)
    ball_triangles[2::6, 0] = sin_phi_dp * cos_theta_dt
    ball_triangles[2::6, 1] = sin_phi_dp * sin_theta_dt
    ball_triangles[2::6, 2] = cos_phi_dp

    # Vertex 3: (phi, theta) - same as vertex 0
    ball_triangles[3::6, 0] = sin_phi * cos_theta
    ball_triangles[3::6, 1] = sin_phi * sin_theta
    ball_triangles[3::6, 2] = cos_phi

    # Vertex 4: (phi + dphi, theta + dtheta) - same as vertex 2
    ball_triangles[4::6, 0] = sin_phi_dp * cos_theta_dt
    ball_triangles[4::6, 1] = sin_phi_dp * sin_theta_dt
    ball_triangles[4::6, 2] = cos_phi_dp

    # Vertex 5: (phi + dphi, theta)
    ball_triangles[5::6, 0] = sin_phi_dp * cos_theta
    ball_triangles[5::6, 1] = sin_phi_dp * sin_theta
    ball_triangles[5::6, 2] = cos_phi_dp

    return ball_triangles


def generategltffiles(filenameroot, rodsstates, ballsstates, ballradiuses, config):
    print("calculating visualization..", end="")
    simsteps = len(rodsstates)

    # Generate ball triangles using vectorized function
    ball_triangles = generate_ball_triangles(config)

    # generate the points of the surface in the format of a triangle strip array
    def generatetrianglestripsurface(rods):
        surface_triangles = []

        yrangenoflip = np.linspace(0.0, config.D, config.TRIANGLES, endpoint=False)
        yrangeflip = np.flip(yrangenoflip)
        xrangenoflip = np.linspace(0.0, config.D, config.TRIANGLES, endpoint=True)
        xrangeflip = np.flip(xrangenoflip)

        modulelist = []
        for i in range(config.GRIDSIZEX - 1):
            if i % 2 == 0:
                flipy = True
                flipx = False
                for j in range(config.GRIDSIZEY - 1):
                    modulelist.append((i, j, flipx, flipy))
                    flipx = not flipx
            else:
                flipy = False
                flipx = False
                for j in reversed(range(config.GRIDSIZEY - 1)):
                    modulelist.append((i, j, flipx, flipy))
                    flipx = not flipx

        # assume number of modules per side is even - number of rods is uneven
        for i, j, flipx, flipy in modulelist:
            rod_nw = rods[i][j + 1]
            rod_sw = rods[i][j]
            rod_se = rods[i + 1][j]
            rod_ne = rods[i + 1][j + 1]

            cat_w = cat.findcatenaryparameters(config.LF, config.D, rod_sw[2], rod_nw[2])
            cat_e = cat.findcatenaryparameters(config.LF, config.D, rod_se[2], rod_ne[2])

            dy = float(config.D) / config.TRIANGLES

            localflipx = flipx

            if not flipy:
                yrange = yrangeflip
            else:
                yrange = yrangenoflip

            for y in yrange:
                height_w_y = cat.catenary(y, cat_w)
                height_e_y = cat.catenary(y, cat_e)
                height_w_dy = cat.catenary(y + dy, cat_w)
                height_e_dy = cat.catenary(y + dy, cat_e)

                cat_we_y = cat.findcatenaryparameters(config.LF, config.D, height_w_y, height_e_y)
                cat_we_dy = cat.findcatenaryparameters(config.LF, config.D, height_w_dy, height_e_dy)

                if localflipx:
                    for x in xrangeflip:
                        surface_triangles.append(((x + i * (config.EXPLODE * config.D)), config.EXPLODE * y + dy + j * (config.EXPLODE * config.D), cat.catenary(x, cat_we_dy)))
                        surface_triangles.append(((x + i * (config.EXPLODE * config.D)), config.EXPLODE * y + j * (config.EXPLODE * config.D), cat.catenary(x, cat_we_y)))
                    localflipx = False
                else:
                    for x in xrangenoflip:
                        surface_triangles.append(((x + i * (config.EXPLODE * config.D)), config.EXPLODE * y + j * (config.EXPLODE * config.D), cat.catenary(x, cat_we_y)))
                        surface_triangles.append(((x + i * (config.EXPLODE * config.D)), config.EXPLODE * y + dy + j * (config.EXPLODE * config.D), cat.catenary(x, cat_we_dy)))
                    localflipx = True

        return surface_triangles

    trianglestrips = []
    first = True
    for rods in rodsstates:
        atrianglestrip = np.array(generatetrianglestripsurface(rods))

        if first:
            # Vectorized: scale x,y by 0.5, set z to 0
            basistrianglestrip = atrianglestrip * np.array([0.5, 0.5, 0.0])
            trianglestrips.append(basistrianglestrip.tolist())
            first = False

        # Vectorized: scale all by 0.5
        scaled_array = (atrianglestrip * 0.5).tolist()
        trianglestrips.append(scaled_array)

    # Vectorized: identity matrix for weights
    weightarrays = np.eye(simsteps, dtype=np.float32).tolist()

    # save visualization in the gltf format

    # step 1: change all data to byte format and save it in binary format

    surface_vertices = np.array(trianglestrips, dtype=np.float32)
    surface_vertices_binary_blob = surface_vertices.flatten().tobytes()

    surface_indices = np.arange(len(surface_vertices[0]), dtype=np.uint32)
    surface_indices_binary_blob = surface_indices.tobytes()

    ball_vertices = np.asarray(ball_triangles, dtype=np.float32)
    ball_vertices_binary_blob = ball_vertices.tobytes()

    ball_indices = np.arange(len(ball_vertices), dtype=np.uint32)
    ball_indices_binary_blob = ball_indices.tobytes()

    ball_timestamps = np.linspace(0, len(ballsstates) * config.DT, len(ballsstates), dtype=np.float32)
    ball_timestamps_binary_blob = ball_timestamps.tobytes()

    ball_positions = np.array(ballsstates, dtype=np.float32)
    ball_positions_transposed = ball_positions.transpose(1, 0, 2)
    ball_positions_binary_blob = ball_positions_transposed.tobytes()

    np_weight_arrays = np.array(weightarrays, dtype=np.float32)
    weight_arrays_binary_blob = np_weight_arrays.flatten().tobytes()

    # Save binary buffer to .bin file
    with open(OUTPUT_DIR / (filenameroot + ".bin"), "wb") as f:
        f.write(surface_indices_binary_blob)
        f.write(surface_vertices_binary_blob)
        f.write(ball_indices_binary_blob)
        f.write(ball_vertices_binary_blob)
        f.write(ball_timestamps_binary_blob)
        f.write(ball_positions_binary_blob)
        f.write(weight_arrays_binary_blob)

    # step 2: generate the gltfobj file and point it to the data and save it
    gltfobj = gltf.GLTF2()
    gltfobj.scene = 0
    scene = gltf.Scene()

    # add nodes
    # camera
    camera_matrix = [
        0.996529757976532,
        0,
        -0.08323691785335541,
        0,
        0.056379012763500214,
        0.7356777191162109,
        0.6749812960624695,
        0,
        0.06123554706573486,
        -0.677331805229187,
        0.7331247329711914,
        0,
        3.3165924549102783,
        -3.3049261569976807,
        6.134603023529053,
        1
    ]

    gltfobj.cameras.append(gltf.Camera(type="orthographic", orthographic=gltf.Orthographic(
        xmag=1.0,
        ymag=1.0,
        zfar=56.84701458464913,
        znear=0.005684701458464913)))

    gltfobj.nodes.append(gltf.Node(camera=0, matrix=camera_matrix))
    scene.nodes.append(0)

    # robot
    gltfobj.nodes.append(gltf.Node(mesh=1, scale=[1.0, 1.0, 2.0]))  # somehow the triangle array is shown in 2x scale so we have to scale down
    scene.nodes.append(1)

    # balls
    for idx, r in enumerate(ballradiuses, start=2):
        gltfobj.nodes.append(gltf.Node(mesh=0, scale=[r, r, r]))
        scene.nodes.append(idx)

    # add nodes to scene
    gltfobj.scenes.append(scene)

    gltfobj.materials.append(
        gltf.Material(
            pbrMetallicRoughness=gltf.PbrMetallicRoughness(
                baseColorFactor=[1.0, 0.0, 0.0, 1.0],  # Red
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
            doubleSided=True,
            alphaMode="MASK",
        )
    )

    gltfobj.materials.append(
        gltf.Material(
            pbrMetallicRoughness=gltf.PbrMetallicRoughness(
                baseColorFactor=[0.0, 1.0, 0.0, 1.0],  # Black
                metallicFactor=0.0,
                roughnessFactor=1.0
            ),
            alphaMode="MASK",
            doubleSided=True,
        )
    )

    # ball mesh (reused)
    gltfobj.meshes.append(
        gltf.Mesh(
            primitives=[
                gltf.Primitive(
                    attributes=gltf.Attributes(POSITION=1), indices=0, material=1, mode=gltf.TRIANGLES
                )
            ]
        )
    )

    # Vectorized: create targets list
    thetargets = [gltf.Attributes(POSITION=6 + i) for i in range(simsteps)]

    # surface
    gltfobj.meshes.append(
        gltf.Mesh(
            primitives=[
                gltf.Primitive(
                    attributes=gltf.Attributes(POSITION=4), indices=3, material=0, mode=gltf.TRIANGLE_STRIP, targets=thetargets
                )
            ],
            weights=weightarrays[0],
        )
    )

    # animation
    # surface sampler
    thesamplers = [gltf.AnimationSampler(input=2, interpolation="LINEAR", output=5)]
    # ball samplers
    thesamplers.extend([
        gltf.AnimationSampler(input=2, interpolation="LINEAR", output=i)
        for i in range(6 + simsteps, 6 + simsteps + len(ballradiuses))
    ])

    # channels
    thechannels = [gltf.AnimationChannel(sampler=0, target=gltf.AnimationChannelTarget(node=1, path="weights"))]
    thechannels.extend([
        gltf.AnimationChannel(sampler=i + 1, target=gltf.AnimationChannelTarget(node=i + 2, path="translation"))
        for i in range(len(ballradiuses))
    ])

    gltfobj.animations.append(gltf.Animation(samplers=thesamplers, channels=thechannels))

    # 0 - ball - triangle indices
    gltfobj.accessors.append(
        gltf.Accessor(
            bufferView=2,
            componentType=gltf.UNSIGNED_INT,
            count=ball_indices.size,
            type=gltf.SCALAR,
            max=[int(ball_indices.max())],
            min=[int(ball_indices.min())],
        )
    )

    # 1 - ball - vertices
    gltfobj.accessors.append(
        gltf.Accessor(
            bufferView=3,
            componentType=gltf.FLOAT,
            count=len(ball_vertices),
            type=gltf.VEC3,
            max=ball_vertices.max(axis=0).tolist(),
            min=ball_vertices.min(axis=0).tolist(),
        )
    )

    # 2 - timesteps
    gltfobj.accessors.append(
        gltf.Accessor(
            bufferView=4,
            componentType=gltf.FLOAT,
            count=len(ballsstates),
            type=gltf.SCALAR,
            max=[len(ballsstates) * config.DT],
            min=[0.0],
        )
    )

    # 3 - surface indicies
    gltfobj.accessors.append(
        gltf.Accessor(
            bufferView=0,
            componentType=gltf.UNSIGNED_INT,
            count=len(surface_indices),
            type=gltf.SCALAR,
            max=[int(surface_indices.max())],
            min=[0],
        )
    )

    # 4 - surface vertices
    gltfobj.accessors.append(
        gltf.Accessor(
            bufferView=1,
            componentType=gltf.FLOAT,
            count=len(surface_vertices[0]),
            type=gltf.VEC3,
            max=surface_vertices[0].max(axis=0).tolist(),
            min=surface_vertices[0].min(axis=0).tolist(),
        )
    )

    # 5 - weights
    gltfobj.accessors.append(
        gltf.Accessor(
            bufferView=6,
            componentType=gltf.FLOAT,
            count=simsteps * simsteps,
            type=gltf.SCALAR,
            max=[1.0],
            min=[0.0],
        )
    )

    # 6 to 6 + simsteps - 1: surface vertices per frame
    for i in range(simsteps):
        gltfobj.accessors.append(
            gltf.Accessor(
                bufferView=1,
                componentType=gltf.FLOAT,
                byteOffset=len(surface_vertices[0]) * 3 * 4 * i,
                count=len(surface_vertices[0]),
                type=gltf.VEC3,
                max=surface_vertices[i].max(axis=0).tolist(),
                min=surface_vertices[i].min(axis=0).tolist(),
            )
        )

    # 6 + simsteps to 6 + simsteps + number of balls - 1: ball positions
    for i in range(len(ball_positions[0])):
        gltfobj.accessors.append(
            gltf.Accessor(
                bufferView=5,
                componentType=gltf.FLOAT,
                byteOffset=simsteps * 3 * 4 * i,
                count=simsteps,
                type=gltf.VEC3,
                max=ball_positions_transposed[i].max(axis=0).tolist(),
                min=ball_positions_transposed[i].min(axis=0).tolist(),
            )
        )

    # views of the data in the binary file
    gltfobj.bufferViews.append(
        gltf.BufferView(
            buffer=0,
            byteLength=len(surface_indices_binary_blob),
            target=gltf.ELEMENT_ARRAY_BUFFER,
        )
    )
    gltfobj.bufferViews.append(
        gltf.BufferView(
            buffer=0,
            byteStride=12,
            byteOffset=len(surface_indices_binary_blob),
            byteLength=len(surface_vertices_binary_blob),
            target=gltf.ARRAY_BUFFER,
        )
    )
    gltfobj.bufferViews.append(
        gltf.BufferView(
            buffer=0,
            byteOffset=len(surface_indices_binary_blob) + len(surface_vertices_binary_blob),
            byteLength=len(ball_indices_binary_blob),
            target=gltf.ELEMENT_ARRAY_BUFFER,
        )
    )
    gltfobj.bufferViews.append(
        gltf.BufferView(
            buffer=0,
            byteOffset=len(surface_indices_binary_blob) + len(surface_vertices_binary_blob) + len(ball_indices_binary_blob),
            byteLength=len(ball_vertices_binary_blob),
            target=gltf.ARRAY_BUFFER,
        )
    )
    gltfobj.bufferViews.append(
        gltf.BufferView(
            buffer=0,
            byteOffset=len(surface_indices_binary_blob) + len(surface_vertices_binary_blob) + len(ball_indices_binary_blob) + len(ball_vertices_binary_blob),
            byteLength=len(ball_timestamps_binary_blob)
        )
    )
    gltfobj.bufferViews.append(
        gltf.BufferView(
            buffer=0,
            byteOffset=len(surface_indices_binary_blob) + len(surface_vertices_binary_blob) + len(ball_indices_binary_blob) + len(ball_vertices_binary_blob) + len(ball_timestamps_binary_blob),
            byteLength=len(ball_positions_binary_blob),
        )
    )

    gltfobj.bufferViews.append(
        gltf.BufferView(
            buffer=0,
            byteOffset=len(surface_indices_binary_blob) + len(surface_vertices_binary_blob) + len(ball_indices_binary_blob) + len(ball_vertices_binary_blob) + len(ball_timestamps_binary_blob) + len(ball_positions_binary_blob),
            byteLength=len(weight_arrays_binary_blob),
        )
    )

    gltfobj.buffers.append(
        gltf.Buffer(
            byteLength=len(surface_indices_binary_blob) + len(surface_vertices_binary_blob) + len(ball_indices_binary_blob) + len(ball_vertices_binary_blob) + len(ball_timestamps_binary_blob) + len(ball_positions_binary_blob) + len(weight_arrays_binary_blob)
        )
    )

    # Link buffer URI
    gltfobj.buffers[0].uri = filenameroot + ".bin"

    # Save glTF file
    gltfobj.save(str(OUTPUT_DIR / (filenameroot + ".gltf")))

    print(filenameroot + " gltf files written")
