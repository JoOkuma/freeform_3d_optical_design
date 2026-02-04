import time
import numpy as np
from ray_propagation import (
    Coordinates, Refractive3dOptic, FixedIndexMaterial,
    TrainingData_for_2dImaging, from_tif, to_tif,
    plot_loss_history, plot_simple_ray_diagram)

def example_of_usage():
    """Example code: design a 3D refractive optic with specified input/output.

    Consider copy-pasting this example code to get you started.

    In this example, the input/output is simple plane-to-plane imaging
    (with inversion). This is the same input-output you'd expect from a
    pair of ideal lenses which are cofocal and coaxial.

    We start with some (suboptimal) 3D refractive optic, and we generate
    "training data": bundles of rays that represent the position and
    direction of optical inputs to our 3D optic. For each input, we
    specify the output that we WISH our optic would deliver, and then
    calculate the output it ACTUALLY delivers, for our current 3D
    refractive optic. We use the difference between desired and
    calculated output to calculate our "loss", and use gradients of this
    loss to update our 3D refractive optic.
    """

    # Specify our coordinate system, organized via a Coordinates object:
    coords = Coordinates(xyz_i=(-10.0, -10.0,   0.0),
                         xyz_f=(+10.0, +10.0, +20.0),
                         n_xyz=(  101,   101,   101))
    print("Voxel size: %0.3f, %0.3f, %0.3f"%(coords.dx, coords.dy, coords.dz))

    # Use these coordinates to initialize an instance of Refractive3dOptic
    # that will simulate how light changes as it passes through our
    # refractive optic:
    ro = Refractive3dOptic(coords)

    # Each voxel of our refractive optic is a mixture of materials:
    air     = FixedIndexMaterial(1)
    polymer = FixedIndexMaterial(1.5)
    ro.set_materials((air, polymer))

    # Initialize our optic.
    try: # If there's a concentration saved to disk, pick up where we left off:
        fname = '01_concentration.tif'
        initial_concentration = from_tif(fname)
        ro.set_3d_concentration(initial_concentration)
        print("Using initial concentration from:", fname)
    except FileNotFoundError:
        print("Using default concentration (50/50 mixture at each voxel).")

    # Make a source to generate training data. In this case, the
    # training data is for a simple plane-to-plane inverting imaging
    # system:
    data_source = TrainingData_for_2dImaging(
        coords, radius=3, max_z_angle_deg=15, x0=0, y0=0, num_rays=1000)
    loss_history = []
    for iteration in range(int(1e6)): # Run for a loooong time
        start_time = time.perf_counter()
        
        # Use our data source to generate random input/output pairs:
        wavelength = 0.5
        input_rays, desired_output_rays = (
            data_source.input_output_pair(wavelength))
        ro.set_input_raybundle(input_rays)
        ro.set_desired_output_raybundle(desired_output_rays)

        # Simulate propagation through our 3D refractive optic,
        # calculate loss, and calculate a gradient that hopefully will
        # reduce the loss:
        from profilehooks import profile
        profile(ro.gradient_update, immediate=True)(
            dt=0.1,
            step_size=10,
            z_planes=(0, 1),
            smoothing_sigma=5,
        )
        x0, y0 = 0, 0 # TODO: remove this cruft
        loss_history.append((x0, y0, ro.loss))


        import torch
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
            ro.gradient_update(
                dt=0.1,
                step_size=10,
                z_planes=(0, 1),
                smoothing_sigma=5)
            x0, y0 = 0, 0 # TODO: remove this cruft
            loss_history.append((x0, y0, ro.loss))

        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

        end_time = time.perf_counter()
        print("At iteration", iteration, "the loss is %0.4f"%(ro.loss),
              "(%0.2f ms elapsed)"%(1000*(end_time - start_time)))

        # Every so often, output some intermediate state, so we can
        # monitor our progress. You can use ImageJ
        # ( https://imagej.net/ij/ ) to view the TIF files:
        if iteration % 10 == 0:
            ro.update_attributes()
            print("Saving TIFs etc...", end='')
            to_tif('00_composition.tif',          ro.composition)
            to_tif('01_concentration.tif',        ro.concentration)
            to_tif('02_concentration_xz.tif',
                   ro.concentration[:, ro.coordinates.ny//2, :])
            to_tif('03_gradient.tif', ro.gradient)
            plot_simple_ray_diagram(ro, '04_ray_diagram.png')
            plot_loss_history(loss_history, '05_loss_history.png')
            print("done.")

if __name__ == '__main__':
    example_of_usage()
