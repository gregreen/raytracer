#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


## Numeric identifiers for different object types
#PLANE = 0
#SPHERE = 1

# Standard float datatype
FLOAT_DTYPE = 'f8'
# TODO: Make sure all instances of np.zeros, np.empty, np.array, etc. use
# this datatype. It may also be necessary to call ().astype(FLOAT_DTYPE).


def plane_intersection(x0, v, p0, n):
    """
    Computes the intersections of a set of rays with a set of planes.

    Inputs:
        x0 (np.ndarray): Coords of ray origin. Shape = (# of rays, # of dims).
        v (np.ndarray): Direction of ray. Shape = (# of rays, # of dims).
        p0 (np.ndarray): Point on plane. Shape = (# of planes, # of dims).
        n (np.ndarray): Normal to plane. Shape = (# of planes, # of dims).

    Outputs:
        t (np.ndarray): Intersection dist. Shape = (# of rays, # of planes).
    """
    num = np.sum((p0[None,:,:]-x0[:,None,:])*n[None,:,:], axis=2)
    denom = np.sum(v[:,None,:]*n[None,:,:], axis=2)
    return num / denom


def sphere_intersection(x0, v, p0, r, unroll_multiple_intersections=True):
    """
    Computes the intersections of a set of rays with a set of spheres.

    Inputs:
        x0 (np.ndarray): Coords of ray origin. Shape = (# of rays, # of dims).
        v (np.ndarray): Direction of ray. Shape = (# of rays, # of dims).
        p0 (np.ndarray): Sphere center. Shape = (# of spheres, # of dims).
        r (np.ndarray): Sphere radii. Shape = (# of spheres,).

    Outputs:
        t (np.ndarray): Intersection dist. Shape = (# rays, # spheres, 2).
    """
    v2 = np.sum(v**2, axis=1) # shape = (ray,)
    # Vector from ray origin to sphere origin
    rho = p0[None,:,:] - x0[:,None,:] # shape = (ray, sphere, dim)
    # Distance^2 from ray origin to sphere origin: (p0 - x0)^2
    rho2 = np.sum(rho*rho, axis=2) # shape = (ray, sphere)
    # (p0 - x0) * v
    eta = np.sum(rho*v[:,None,:], axis=2) # shape = (ray, sphere)
    # Discriminant (in quadratic equation): eta^2 - v^2 * (rho^2 - r^2)
    delta2 = eta**2 - v2[:,None]*(rho2-(r**2)[None,:]) # shape = (ray, sphere)
    delta = np.sqrt(delta2)
    # Two different intersection points (could be the same or imaginary)
    n_rays = x0.shape[0]
    n_spheres = p0.shape[0]
    t = np.empty((n_rays, n_spheres, 2), dtype=FLOAT_DTYPE)
    t[:,:,0] = (eta + delta) / v2[:,None]
    t[:,:,1] = (eta - delta) / v2[:,None]
    # Replace NaNs with infs
    t[~np.isfinite(t)] = np.inf
    # Unroll last dimension (the intersection dimension)?
    if unroll_multiple_intersections:
        t.shape = (n_rays, 2*n_spheres)
    return t


def sphere_normal(x0, p0):
    """
    Computes the normal to a sphere at the point x0 (assumed to be on the
    sphere, though this is not verified.

    Inputs:
        x0 (np.ndarray): Coords of ray origin. Shape = (# spheres, # dims).
        p0 (np.ndarray): Sphere center. Shape = (# spheres, # dims).

    Outputs:
        n (np.ndarray): Normal vectors. Shape = (# spheres, # dims).
    """
    rho = p0 - x0
    rho /= np.linalg.norm(rho, axis=1)[:,None]
    return rho


def find_closest_intersections(t, eps=1e-10):
    t[t<eps] = np.inf
    idx = np.argmin(t, axis=1)
    return idx, t[np.arange(len(idx)),idx]


def plot_scene(rays, scene, recursion_depth=3):
    x0 = rays['x0']
    v = rays['v']

    n_dim = x0.shape[1]

    # Get ray colors
    ray_value, ray_props = render_rays_recursive(
        recursion_depth,
        x0, v,
        scene,
        collect_rays=True
    )

    for key in ray_props:
        ray_props[key] = np.concatenate(ray_props[key], axis=0)

    #print(ray_value)
    #print(ray_props)

    # Figure
    fig,ax = plt.subplots(
        1,1, figsize=(6,6),
        subplot_kw=dict(aspect='equal')
    )

    # Draw planes
    p0 = scene['planes']['p0']
    pn = scene['planes']['n']
    pc = scene['planes']['color']
    pr = scene['planes']['reflectivity']
    for pp,nn,cc,rr in zip(p0,pn,pc,pr):
        ax.scatter([pp[0]], [pp[1]], color=cc)
        ax.arrow(
            pp[0], pp[1],
            nn[0], nn[1],
            color=cc,
            head_width=0.2
        )
        ax.plot(
            [pp[0]-30*nn[1], pp[0]+30*nn[1]],
            [pp[1]+30*nn[0], pp[1]-30*nn[0]],
            color=cc,
            alpha=np.mean(rr),
            lw=3
        )

    # Draw spheres
    s0 = scene['spheres']['p0']
    srad = scene['spheres']['r']
    sc = scene['spheres']['color']
    srefl = scene['spheres']['reflectivity']
    for pp,rad,cc,refl in zip(s0,srad,sc,srefl):
        ax.add_patch(patches.Circle(
            pp, radius=rad,
            edgecolor=cc,
            facecolor='none',
            alpha=np.mean(refl),
            lw=3.
        ))

    # Draw intersections
    c_norm = np.max(ray_props['ray_value'])
    idx = np.isfinite(ray_props['t'])
    xp = (
        ray_props['x0'][idx,:]
      + ray_props['v'][idx,:]*ray_props['t'][idx,None]
    )
    for xx,xo,vv,cc in zip(xp, ray_props['x0'][idx],
                           ray_props['v'][idx],
                           ray_props['ray_value'][idx]
                          ):
        ax.scatter([xx[0]], [xx[1]], color=cc/c_norm)
        ax.plot([xo[0],xx[0]],[xo[1],xx[1]], color=cc/c_norm, alpha=0.8)

    # Draw rays that go to infinity
    for xo,vv in zip(ray_props['x0'][~idx], ray_props['v'][~idx]):
        ax.plot(
            [xo[0], xo[0]+30*vv[0]],
            [xo[1], xo[1]+30*vv[1]],
            color='k',
            alpha=0.05
        )

    # Draw camera rays
    ax.scatter(x0[:,0], x0[:,1], c='blue')
    for xx,vv in zip(x0,v):
        ax.arrow(
            xx[0], xx[1],
            0.5*vv[0], 0.5*vv[1],
            color='blue',
            head_width=0.1,
            zorder=10000,
            alpha=0.5
        )

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    return fig


def load_scene(fname):
    import json
    with open(fname, 'r') as f:
        d = json.load(f)

    n_dim = d['n_dim']
    n_channels = d['n_channels']

    # Planes
    for key in ('p0', 'n'):
        d['planes'][key] = np.array(d['planes'][key], dtype=FLOAT_DTYPE)
        if len(d['planes'][key]) == 0:
            d['planes'][key].shape = (0,n_dim)

    # Spheres
    for key in ('p0', 'r'):
        d['spheres'][key] = np.array(d['spheres'][key], dtype=FLOAT_DTYPE)
    if len(d['spheres']['p0']) == 0:
        d['spheres']['p0'].shape = (0,n_dim)

    # Material properties (all object geometries)
    for geom in ('planes', 'spheres'):
        for prop in ('color', 'reflectivity'):
            d[geom][prop] = np.array(d[geom][prop], dtype=FLOAT_DTYPE)
            if len(d[geom][prop]) == 0:
                d[geom][prop].shape = (0,n_channels)

    # Camera
    camera = d.pop('camera')
    camera['shape'] = tuple(camera['shape'])
    camera['v'] = gnomonic_projection(
        camera['fov'],
        camera['shape'],
        flatten=True
    )
    camera['x0'] = np.tile(
        camera['x0'],
        (camera['v'].shape[0], 1)
    )

    return camera, d


def gen_rand_scene(n_dim, camera_shape,
                   n_planes, n_spheres,
                   fov=75., rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Random rays
    #x0 = 0.5 * rng.normal(size=(n_rays, n_dim))
    #v = rng.normal(size=(n_rays, n_dim))
    #v /= np.linalg.norm(v, axis=1)[:,None]

    # Rays: Gnomonic camera
    v = gnomonic_projection(fov, camera_shape, flatten=True)
    x0 = np.zeros_like(v)
    theta = rng.uniform(0., 2*np.pi)
    rotate_vectors(v, 0, 1, theta)

    # Planes
    p0 = 4 * rng.normal(size=(n_planes, n_dim))
    pn = rng.normal(size=(n_planes, n_dim))
    pn /= np.linalg.norm(pn, axis=1)[:,None]

    # Plane colors
    pc = rng.uniform(size=(n_planes, 3))

    # Plane reflectivity
    pr = rng.uniform(0., 0.5, size=(n_planes, 3))

    # Choose some planes to have no color of their own:
    idx = np.arange(n_planes)
    rng.shuffle(idx)
    idx = idx[:n_planes//2]
    pc[idx] *= 0.1

    # Spheres
    s0 = 4 * rng.normal(size=(n_spheres, n_dim))
    r = rng.uniform(0.5, 2., size=n_spheres)

    # Sphere colors
    sc = rng.uniform(size=(n_spheres, 3))

    # Sphere reflectivity
    sr = rng.uniform(0., 0.5, size=(n_spheres, 3))

    rays = {
        'x0': x0.astype(FLOAT_DTYPE),
        'v': v.astype(FLOAT_DTYPE)
    }
    scene = {
        'n_dim': n_dim,
        'n_channels': 3,
        'planes': {
            'p0': p0.astype(FLOAT_DTYPE),
            'n': pn.astype(FLOAT_DTYPE),
            'color': sc.astype(FLOAT_DTYPE),
            'reflectivity': sr.astype(FLOAT_DTYPE)
            # Later, diffusivity, specularity, transmission, ind. of refr.
        },
        'spheres': {
            'p0': s0.astype(FLOAT_DTYPE),
            'r': r.astype(FLOAT_DTYPE),
            'color': sc.astype(FLOAT_DTYPE),
            'reflectivity': sr.astype(FLOAT_DTYPE)
        }
        # Later, trianges, etc.
    }
    return rays, scene
    # TODO: Generate camera and scene separately


def mirror_reflection_outgoing(vi, n):
    """
    Computes the direction of an outgoing ray generated by a mirror
    reflection.

    Inputs:
        vi (np.ndarray): Direction of incoming ray.
                         Shape = (# of rays, # of dims).
        n (np.ndarray): Normal to surface. Shape = (# of rays, # of dims).

    Outputs:
        vo (np.ndarray): Direction of outgoing ray.
                         Shape = (# of rays, # of dims).
    """
    return vi - 2 * n * np.sum(vi*n, axis=1)[:,None]

def render_rays_recursive(
                          recursion_limit,
                          x0, v, # Ray properties
                          scene, # Properties of objects in scene
                          collect_rays=False,
                          #ray_parent_id=None,
                          #ray_contribution=None,
                          recursion_depth=0
                         ):
    """
    Computes the value of each ray, recursively spawning child rays
    from interactions with surfaces. Rays will be generated to a given
    recursion depth, and then their contributions to the original rays will
    be summed.

    Inputs:
        recursion_limit (int): Depth at which recursion is truncated.
            Effectively, the max number of scatterings, refractions,
            etc. that each ray can make.
        x0 (np.ndarray): Coords of ray origin. Shape = (# of rays, # of dims).
        v (np.ndarray): Direction of ray. Shape = (# of rays, # of dims).
        p0 (np.ndarray): Point on plane. Shape = (# of planes, # of dims).
        pn (np.ndarray): Normal to plane. Shape = (# of planes, # of dims).
        pc (np.ndarray): Normal to plane.
            Shape = (# of planes, # of channels).
        pr (np.ndarray): Plane reflectivity.
            Shape = (# of planes, # of channels).
        collect_rays (Optional[bool]): If `True`, all rays at all depths
            will be returned. Defaults to `False`.
        recursion_depth (Optional[int]): Depth of recursion. Top level
            has depth of 0.
        ray_parent_id (Optional[np.ndarray]): ID of top-level
            parent (could be a screen pixel index). Shape = (# of rays,).
        ray_contribution (Optional[np.ndarray]): Total contribution of
            the ray to the original pixel, in each channel. Top-level rays
            will have a contribution of 1 (in each channel), while rays
            at each subsequent recursion depth will have progressively
            smaller contributions. Shape = (# of rays, # of channels).

    Outputs:
        ray_value (np.ndarray): Color value of each ray.
            Shape = (# of rays, # of channels).
        x0_all_rays (np.ndarray): Starting point of each ray, at all
            recursion depths. Only returned if `collect_rays` is `True`.
        v_all_rays (np.ndarray): Direction of each ray, at all recursion
            depths. Only returned if `collect_rays` is `True`.
        t_all_rays (np.ndarray): Distance to next intersection of each ray,
            at all recursion depths. Only returned if `collect_rays`
            is `True`.
        value_all_rays (np.ndarray): Value of each ray, at all recursion
            depths. Only returned if `collect_rays` is `True`.
    """
    n_rays, n_dim = x0.shape
    n_channels = scene['n_channels']

    ray_value = np.zeros((n_rays, n_channels), dtype=FLOAT_DTYPE)

    # Empty array for all intersections
    n_planes = len(scene['planes']['p0'])
    n_spheres = len(scene['spheres']['p0'])
    t = np.empty((n_rays,n_planes+2*n_spheres), dtype=FLOAT_DTYPE)

    # Plane intersections. Shape = (# of rays, # of planes)
    t[:,:n_planes] = plane_intersection(
        x0, v,
        scene['planes']['p0'],
        scene['planes']['n']
    )

    # Sphere intersections. Shape = (# of rays, # of spheres)
    t[:,n_planes:] = sphere_intersection(
        x0, v,
        scene['spheres']['p0'],
        scene['spheres']['r'],
        unroll_multiple_intersections=True
    )

    # For each ray, calculate closest intersections
    close_idx, t_close = find_closest_intersections(t)
    print('close_idx', close_idx)
    print('t_close', t_close)

    # Identify rays with an intersection (i.e., t not infinite)
    ray_idx = np.where(np.isfinite(t_close))[0]
    obj_idx = close_idx[ray_idx]
    #plane_idx = close_idx[ray_idx]
    print('ray_idx', ray_idx)
    print(f'ray_idx.shape = {ray_idx.shape}')
    print(f'obj_idx.shape = {obj_idx.shape}')

    # Determine what type of object each ray intersects, and determine
    # the index (ID) of that object in its respective object array.
    idx_is_plane = obj_idx < n_planes
    idx_is_sphere = ~idx_is_plane
    print('idx_is_plane', idx_is_plane)
    print('idx_is_sphere', idx_is_sphere)

    plane_id = obj_idx[idx_is_plane]
    sphere_id = (obj_idx[idx_is_sphere] - n_planes) // 2
    print('plane_id', plane_id)
    print('sphere_id', sphere_id)
    #plane_id = np.where(idx_is_plane)[0]
    #sphere_id = (np.where(idx_is_sphere)[2] - n_planes) // 2

    # Add luminosity from sources that are directly hit
    ray_idx_plane = ray_idx[idx_is_plane]
    ray_idx_sphere = ray_idx[idx_is_sphere]
    #print('ray_idx_plane', ray_idx_plane)
    #print('ray_idx_sphere', ray_idx_sphere)
    ray_value[ray_idx_plane] = scene['planes']['color'][plane_id]
    ray_value[ray_idx_sphere] = scene['spheres']['color'][sphere_id]

    if (recursion_depth >= recursion_limit) or (len(t_close) == 0):
        if collect_rays:
            ray_props = {
                'x0': [x0],
                'v': [v],
                't': [t_close],
                'ray_value': [ray_value]
            }
            return ray_value, ray_props
        return ray_value
    
    x0_child = []
    v_child = []
    child_contrib = []
    child_parent_idx = []

    # Calculate intersection coordinates
    x_i = x0[ray_idx] + v[ray_idx] * t_close[ray_idx][:,None]

    # Determine intersection normals
    n_intersects = ray_idx.shape[0]
    n = np.empty(shape=(n_intersects,n_dim), dtype=FLOAT_DTYPE)

    # Empty reflectivity array
    reflect = np.empty(shape=(n_intersects,n_channels), dtype=FLOAT_DTYPE)

    # Plane normals, reflectivity, etc.
    n[idx_is_plane] = scene['planes']['n'][plane_id]
    reflect[idx_is_plane] = scene['planes']['reflectivity'][plane_id]

    # Sphere normals, reflectivity, etc.
    n[idx_is_sphere] = sphere_normal(
        x_i[idx_is_sphere],
        scene['spheres']['p0'][sphere_id]
    )
    reflect[idx_is_sphere] = scene['planes']['reflectivity'][sphere_id]

    # Spawn mirror reflection at each intersection
    x0_child.append(x_i)
    v_child.append(mirror_reflection_outgoing(v[ray_idx], n))
    child_contrib.append(reflect)
    child_parent_idx.append(ray_idx)
    #child_parent_idx.append(np.where(ray_idx)[0])

    # Combine all types of child rays into one array
    x0_child = np.concatenate(x0_child, axis=0)
    v_child = np.concatenate(v_child, axis=0)
    child_contrib = np.concatenate(child_contrib, axis=0)
    child_parent_idx = np.concatenate(child_parent_idx, axis=0)

    #print(f'recursion depth: {recursion_depth}')
    #print('x0:\n', x0_child)
    #print('v:\n', v_child)
    #print('')

    # Recursion: Add in values of spawned rays
    ret = render_rays_recursive(
        recursion_limit,
        x0_child, v_child,
        scene,
        collect_rays=collect_rays,
        recursion_depth=recursion_depth+1
    )
    if collect_rays:
        ray_value_ret, ray_props_ret = ret
    else:
        ray_value_ret = ret

    print(f'ray_value.shape = {ray_value.shape}')
    print(f'child_parent_idx.shape = {child_parent_idx.shape}')
    print(f'child_contrib.shape = {child_contrib.shape}')
    print(f'ray_value_ret.shape = {ray_value_ret.shape}')
    np.add.at(ray_value, child_parent_idx, child_contrib*ray_value_ret)

    if collect_rays:
        ray_props = {
            'x0': [x0],
            'v': [v],
            't': [t_close],
            'ray_value': [ray_value]
        }
        for key in ray_props_ret:
            ray_props[key] += ray_props_ret[key]
        return ray_value, ray_props
    return ray_value


def render_rays(x0, v, p0, n, c):
    # Calculate closest intersections
    t = plane_intersection(x0, v, p0, n)
    close_idx, t_close = find_closest_intersections(t)

    # Identify rays with an intersection
    has_intersection = np.isfinite(t_close)

    ray_idx = has_intersection
    plane_idx = close_idx[has_intersection]

    # Color each ray by surface it hits
    n_rays = x0.shape[0]
    n_channels = c.shape[1]
    ray_value = np.zeros((n_rays, n_channels), dtype=FLOAT_DTYPE)
    ray_value[ray_idx] = c[plane_idx]

    return ray_value


def gnomonic_projection(fov, shape, flatten=False):
    r = 0.5 * shape[0] / np.tan(np.radians(fov))
    screen_coords = np.indices(shape).astype(FLOAT_DTYPE)
    #screen_coords = np.flip(screen_coords, axis=0)
    for i,s in enumerate(shape):
        screen_coords[i] -= 0.5 * (s-1)
    screen_coords = np.concatenate(
        [screen_coords,np.full((1,)+shape,r,dtype=FLOAT_DTYPE)],
        axis=0
    )
    screen_coords /= np.linalg.norm(screen_coords, axis=0)[None]
    screen_coords = np.moveaxis(screen_coords, 0, -1)
    if flatten:
        screen_coords.shape = (-1, screen_coords.shape[-1])
    return screen_coords


def rotate_vectors(x, dim0, dim1, theta):
    y = x[...,(dim0,dim1)]
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    y = np.tensordot(y,R, axes=(-1,1))
    x[...,(dim0,dim1)] = y


def main():
    #x = np.arange(12)
    #x.shape = (4,3)
    #print(x)
    #rotate_vectors(x, 0, 1, np.pi/4.)
    #return 0

    # Generate random scene
    #rng = np.random.default_rng(4)
    #n_dim = 2
    #camera_shape = (20,)
    #n_planes = 4
    #n_spheres = 3
    ##x0,v,p0,pn,pc,pr = gen_rand_scene(n_dim, camera_shape, n_planes, rng=rng)
    #camera, scene = gen_rand_scene(
    #    n_dim, camera_shape,
    #    n_planes, n_spheres,
    #    rng=rng
    #)
    #print(scene)

    camera, scene = load_scene('box_with_light.json')
    n_dim = scene['n_dim']
    camera_shape = camera['shape']
    x0 = camera['x0']
    v = camera['v']
    print(scene)

    # Plot scene
    if n_dim == 2:
        fig = plot_scene(camera, scene, recursion_depth=3)
        fig.savefig('ray_diagram_2d.svg')
        #plt.show()

    # Render scene
    from tqdm import tqdm
    n_frames = 360

    for max_depth in range(4,5):
        print(f'Rendering scene at max depth {max_depth} ...')
        v_rot = camera['v'].copy()

        for frame in tqdm(range(n_frames)):
            pixel_color = render_rays_recursive(
                max_depth,
                camera['x0'], v_rot,
                scene,
                collect_rays=False
            )
            pixel_color /= np.max(pixel_color)
            if n_dim == 2:
                pixel_color.shape = (1,)+pixel_color.shape
            elif n_dim == 3:
                pixel_color.shape = camera_shape + (3,)
                pixel_color = np.swapaxes(pixel_color, 0, 1)
            pixel_color = np.clip(255*pixel_color,0.,255.).astype('u1')
            #print(pixel_color.shape)
            im = Image.fromarray(pixel_color, mode='RGB')
            im.save(
                'frames/rendered_scene'
                f'_maxdepth{max_depth}'
                f'_frame{frame:05d}'
                '.png'
            )

            rotate_vectors(v_rot, 2, 0, 2*np.pi/n_frames)
            #print(v_rot)
            #print(pixel_color)

    return 0

if __name__ == '__main__':
    main()

