import taichi as ti
import math
import time
import random
import numpy as np
from plyfile import PlyData, PlyElement
import os
import utils
from utils import create_output_folder
from engine.mpm_solver import MPMSolver

with_gui = True
write_to_disk = False

# Try to run on GPU
ti.init(arch=ti.cuda,
        kernel_profiler=True,
        use_unified_memory=False,
        device_memory_fraction=0.7)

max_num_particles = 400000

if with_gui:
    gui = ti.GUI("MLS-MPM", res=512, background_color=0x112F41)

if write_to_disk:
    output_dir = create_output_folder('./sim')


def visualize(particles):
    np_x = particles['position'] / 1.0

    # simple camera transform
    screen_x = np_x[:, 0]
    screen_y = np_x[:, 1]

    screen_pos = np.stack([screen_x, screen_y], axis=-1)

    gui.circles(screen_pos, radius=0.8, color=particles['color'])
    gui.show()


# Use 512 for final simulation/render
R = 64

mpm = MPMSolver(res=(R, R, R), size=1, unbounded=True, dt_scale=0.5, E_scale=8, max_num_particles=max_num_particles)

mpm.add_surface_collider(point=(0, 0, 0),
                         normal=(0, 1, 0),
                         surface=mpm.surface_slip,
                         friction=3.5)

mpm.set_gravity((0, -25, 0))
counter = 0

start_t = time.time()

addParticlesCount = 2000

for frame in range(15000):
    print(f'frame {frame}')
    t = time.time()

    # mpm.add_cube((0.0, 0.3, 0.0), (0.99999, 0.01, 0.8),
    #              mpm.material_water,
    #              sample_density=10,
    #              color=0x8888FF,
    #              velocity=[0, -1, 0])

    if mpm.n_particles[None] < max_num_particles:
        particles = np.zeros((addParticlesCount, 3), dtype=np.float32)
        for part in range(addParticlesCount):
            particles[part][0] = random.random()
            particles[part][1] = 0.8
            particles[part][2] = random.random()
        mpm.add_particles(particles=particles, material=MPMSolver.material_water, color=0xFF8888, velocity=(0, 0, 0))

    mpm.step(4e-3, print_stat=True)
    if with_gui:
        particles = mpm.particle_info()
        visualize(particles)

    if write_to_disk:
        mpm.write_particles(f'{output_dir}/{frame:05d}.npz')
    print(f'Frame total time {time.time() - t:.3f}')
    print(f'Total running time {time.time() - start_t:.3f}')
