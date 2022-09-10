#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
from scipy.spatial.distance import cdist

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os


## Numeric identifiers for different object types
#PLANE = 0
#SPHERE = 1

# Standard float datatype
FLOAT_DTYPE = 'f4'
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


def triangle_intersection(x0, v, p):
    """
    Computes the intersections of a set of rays with a set of triangles.

    Inputs:
        x0 (np.ndarray): Coords of ray origin. Shape = (# of rays, # of dims).
        v (np.ndarray): Direction of ray. Shape = (# of rays, # of dims).
        p (np.ndarray): Vertices on triangles. Shape = (# triangles, 3, # dims).

    Outputs:
        t (np.ndarray): Intersection dist. Shape = (# rays, # triangles).
    """
    # Compute edge vectors, which connect vertices
    dp = np.diff(p, axis=1, append=p[:,:1,:]) # shape = (triangles, edges, dims)
    # Compute normals
    n = np.cross(dp[:,0], dp[:,1], axisa=1, axisb=1) # shape = (triangles, dims)
    # Compute plane intersections
    t = plane_intersection(x0, v, p[:,0,:], n) # shape = (rays, triangles)
    x1 = x0[:,None,:] + v[:,None,:]*t[:,:,None] # shape = (ray, triangle, dim)
    # Compute edge normals
    b = np.cross(dp, n[:,None,:], axisa=2, axisb=2) # shape = (triangle, edge, dim)
    # Find distance of intersection point from each edge (positive -> outside)
    x1p = x1[:,:,None,:] - p[None,:,:,:] # shape = (ray, triangle, vertex, dim)
    d = np.sum(x1p * b[None,:,:,:], axis=3) # shape = (ray, triangle, edge)
    # Test whether intersections lie inside triangles (d < 0 for every edge)
    idx_outside = np.any(d > 0, axis=2)
    t[idx_outside] = np.inf
    return t


def triangle_normal(p):
    """
    Computes the normals to a set of triangles.

    Inputs:
        p (np.ndarray): Vertices on triangles. Shape = (# triangles, 3, # dims).

    Outputs:
        n (np.ndarray): Normal vectors. Shape = (# triangles, # dims).
    """
    # Compute edge vectors, which connect vertices
    dp = np.diff(p[:,:3], axis=1) # shape = (triangles, edges, dims)
    # Compute normals
    n = np.cross(dp[:,0], dp[:,1], axisa=1, axisb=1) # shape = (triangles, dims)
    return n


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
    sphere, though this is not verified inside the routine.

    Inputs:
        x0 (np.ndarray): Coords of ray origin. Shape = (# spheres, # dims).
        p0 (np.ndarray): Sphere center. Shape = (# spheres, # dims).

    Outputs:
        n (np.ndarray): Normal vectors. Shape = (# spheres, # dims).
    """
    rho = x0 - p0
    # TODO: Could skip norm calculation if r were provided
    rho /= np.linalg.norm(rho, axis=1)[:,None]
    return rho


def find_closest_intersections(t, eps=1e-5 if FLOAT_DTYPE=='f4' else 1e-8):
    t[t<eps] = np.inf
    idx = np.argmin(t, axis=1)
    return idx, t[np.arange(len(idx)),idx]


def plot_scene(rays, scene, recursion_depth=3, rng=None):
    x0 = rays['x0']
    v = rays['v']

    n_dim = x0.shape[1]

    # Get ray colors
    ray_value, ray_props = render_rays_recursive(
        recursion_depth,
        x0, v,
        scene,
        collect_rays=True,
        rng=rng
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
    pr = (
        scene['planes']['reflectivity']
      + scene['planes']['diffusivity']
    ) + 0.1
    pr /= np.max(pr,axis=1)[:,None] + 1.e-8
    for pp,nn,cc,rr in zip(p0,pn,pc,pr):
        ax.scatter([pp[0]], [pp[1]], color=cc)
        ax.arrow(
            pp[0], pp[1],
            nn[0], nn[1],
            color=cc,
            head_width=0.2
        )
        ax.plot(
            [pp[0]-99*nn[1], pp[0]+99*nn[1]],
            [pp[1]+99*nn[0], pp[1]-99*nn[0]],
            color=cc,
            alpha=np.mean(rr),
            lw=3
        )

    # Draw spheres
    s0 = scene['spheres']['p0']
    srad = scene['spheres']['r']
    sc = scene['spheres']['color']# + scene['spheres']['diffusivity']
    #sc /= np.max(sc, axis=1)[:,None] + 1.e-8
    srefl = (
        scene['spheres']['reflectivity']
      + scene['spheres']['diffusivity']
    ) + 0.1
    srefl /= np.max(srefl,axis=1)[:,None] + 1.e-8
    for pp,rad,cc,refl in zip(s0,srad,sc,srefl):
        ax.add_patch(patches.Circle(
            pp, radius=rad,
            edgecolor=cc,
            facecolor='none',
            alpha=np.mean(refl),
            lw=3.
        ))

    # Draw intersections
    c_norm = np.max(ray_props['ray_value']) + 1.e-8
    idx = np.isfinite(ray_props['t'])
    xp = (
        ray_props['x0'][idx,:]
      + ray_props['v'][idx,:]*ray_props['t'][idx,None]
    )
    for xx,xo,vv,cc,nn in zip(xp, ray_props['x0'][idx],
                              ray_props['v'][idx],
                              ray_props['ray_value'][idx],
                              ray_props['n']#[idx]
                             ):
        ax.scatter([xx[0]], [xx[1]], color=cc/c_norm)
        ax.plot([xo[0],xx[0]],[xo[1],xx[1]], color=cc/c_norm, alpha=0.8)
        if np.sum(nn**2) > 1.e-5:
            ax.arrow(
                xx[0], xx[1],
                0.25*nn[0], 0.25*nn[1],
                color='orange',
                head_width=0.05,
                #zorder=10000,
                alpha=0.5
            )

    # Draw rays that go to infinity
    for xo,vv in zip(ray_props['x0'][~idx], ray_props['v'][~idx]):
        ax.plot(
            [xo[0], xo[0]+99*vv[0]],
            [xo[1], xo[1]+99*vv[1]],
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

    # Find most distant object in scene
    x_max = 1.5 * np.max(np.hstack([
        np.abs(scene['planes']['p0']).flat,
        (np.abs(scene['spheres']['p0'])+scene['spheres']['r'][:,None]).flat
    ]))
    xlim = [-x_max, x_max]
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    #ax.set_xlim(-10, 10)
    #ax.set_ylim(-10, 10)

    return fig, ax


def expand_triangle_mesh(verts, faces):
    p = verts[faces]
    return p


def calc_bounding_sphere(v):
    d2_mat = cdist(v, v, metric='sqeuclidean')
    idx_flat = np.argmax(d2_mat)
    idx0,idx1 = np.unravel_index(idx_flat, (v.shape[0],v.shape[0]))
    r = 0.5 * np.sqrt(d2_mat[idx0,idx1])
    x0 = 0.5 * (v[idx0] + v[idx1])
    return x0, r


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

    # Triangles
    d['triangles']['p'] = np.array(d['triangles']['p'], dtype=FLOAT_DTYPE)
    if len(d['triangles']['p']) == 0:
        d['triangles']['p'].shape = (0,3,n_dim)

    # Triangle meshes
    n_triangle_meshes = len(d['triangle_meshes']['vertices'])
    tm = d['triangle_meshes']
    n_tm = len(tm['vertices'])
    d['triangle_meshes']['bounds'] = {
        'x0': np.empty((n_tm,n_dim), dtype=FLOAT_DTYPE),
        'r': np.empty(n_tm, dtype=FLOAT_DTYPE)
    }
    for i in range(n_triangle_meshes):
        tm['vertices'][i] = np.array(tm['vertices'][i], dtype=FLOAT_DTYPE)
        # Max 64k vertices per mesh:
        tm['faces'][i] = np.array(tm['faces'][i], dtype='u2')
        x0,r = calc_bounding_sphere(tm['vertices'][i])
        tm['bounds']['x0'][i] = x0
        tm['bounds']['r'][i] = r

    # Material properties (all object geometries)
    for geom in ('planes', 'spheres', 'triangles', 'triangle_meshes'):
        for prop in ('color', 'reflectivity', 'diffusivity'):
            d[geom][prop] = np.array(d[geom][prop], dtype=FLOAT_DTYPE)
            if len(d[geom][prop]) == 0:
                d[geom][prop].shape = (0,n_channels)
        #print(d[geom]['refract'])
        d[geom]['refract'] = np.array(d[geom]['refract'], dtype=FLOAT_DTYPE)
        if len(d[geom]['refract']) == 0:
            d[geom]['refract'].shape = (0,2)

    # Ambient color
    d['ambient_color'] = np.array(d['ambient_color'])

    # Camera
    camera = d.pop('camera')
    camera['shape'] = tuple(camera['shape'])

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

    # Plane diffusivity
    pd = rng.uniform(0.2, 0.7, size=(n_planes, 3))

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

    # Sphere diffusivity
    sd = rng.uniform(0.2, 0.7, size=(n_spheres, 3))

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
            'color': pc.astype(FLOAT_DTYPE),
            'reflectivity': pr.astype(FLOAT_DTYPE),
            'diffusivity': pd.astype(FLOAT_DTYPE)
            # Later, specularity, transmission, ind. of refr.
        },
        'spheres': {
            'p0': s0.astype(FLOAT_DTYPE),
            'r': r.astype(FLOAT_DTYPE),
            'color': sc.astype(FLOAT_DTYPE),
            'reflectivity': sr.astype(FLOAT_DTYPE),
            'diffusivity': sd.astype(FLOAT_DTYPE)
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


def draw_from_n_sphere_surface(n_vecs, n_dim, rng):
    """
    Draws a random vector from the surface of an n-sphere.

    Inputs:
        n_vecs (int): # of vectors to draw.
        n_dim (int): Dimensionality of space sphere is embedded in.
        rng (np.random.Generator): Numpy pseudorandom bit genenerator.

    Outputs:
        x (np.ndarray): Vector drawn uniformly from the surface of the
            sphere. Shape = (n_vecs, n_dim).
    """
    x = rng.normal(size=(n_vecs, n_dim))
    x /= np.linalg.norm(x, axis=1)[:,None]
    return x


def diffuse_reflection_outgoing(vi, n, rng):
    """
    Generates the direction of an outgoing ray generated by a diffuse
    reflection.

    Inputs:
        vi (np.ndarray): Direction of incoming ray.
            Shape = (# of rays, # of dims).
        n (np.ndarray): Normal to surface. Shape = (# of rays, # of dims).

    Outputs:
        vo (np.ndarray): Direction of outgoing ray.
                         Shape = (# of rays, # of dims).
        cos_phi (np.ndarray): Cosine of angle between incoming ray and
            normal. Used in Lambert's Cosine Law. Shape = (# of rays,).
    """
    n_vecs, n_dim = vi.shape
    vo = draw_from_n_sphere_surface(n_vecs, n_dim, rng)
    vo_dot_n = np.sum(vo*n, axis=1)
    idx_reflect = (np.sign(np.sum(vi*n,axis=1)) == np.sign(vo_dot_n))
    vo[idx_reflect] *= -1
    #vo[idx_reflect] -= 2 * n[idx_reflect] * vo_dot_n[idx_reflect,None]
    vi_norm = np.linalg.norm(vi, axis=1)
    cos_phi = np.abs(vo_dot_n / vi_norm)
    return vo, cos_phi


def refraction_outgoing(vi, n, n1, n2):
    """
    Calculates the directions and strengths of both transmitted and
    reflected rays, using Snell's Law and the Fresnel equations.
    """

    # Calculate direction of transmitted ray using Snell's Law.
    # Formulate as linear combination of incoming and normal vector:
    #   v_t = a*v_i - b*n
    xi = np.sum(vi*n, axis=1)
    idx_flipnormal = (xi > 0) # Normal should not be aligned w/ incoming ray
    xi[idx_flipnormal] *= -1

    a = n1 / n2 # As n2 -> inf, a -> 0, so outgoing ray antialigned w/ normal
    a[idx_flipnormal] = 1 / a[idx_flipnormal]

    delta = a**2 * (xi**2 - 1) + 1
    b = a*xi + np.sqrt(delta)
    b[idx_flipnormal] *= -1

    vt = a[:,None]*vi - b[:,None]*n

    # Reflected ray
    vr = mirror_reflection_outgoing(vi, n)

    # Reflection coefficients for s and p polarizations
    cos_theta_i = -xi
    cos_theta_t = np.abs(np.sum(vt*n, axis=1))
    aa = n1*cos_theta_i
    bb = n2*cos_theta_t
    R_s = np.abs((aa-bb)/(aa+bb))
    aa = n1*cos_theta_t
    bb = n2*cos_theta_i
    R_p = np.abs((aa-bb)/(aa+bb))
    R = 0.5 * (R_s + R_p) # Assume unpolarized light

    # Total (internal) reflection
    idx_reflectall = ~np.isfinite(R)
    R[idx_reflectall] = 1.

    return vt, vr, R


def render_rays_recursive(
                          recursion_limit,
                          x0, v, # Ray properties
                          scene, # Properties of objects in scene
                          collect_rays=False,
                          #ray_parent_id=None,
                          #ray_contribution=None,
                          n_diffuse=4,
                          recursion_depth=0,
                          rng=None
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
    if rng is None:
        rng = np.random.default_rng()

    n_rays, n_dim = x0.shape
    n_channels = scene['n_channels']

    ray_value = np.zeros((n_rays, n_channels), dtype=FLOAT_DTYPE)

    # Empty array for all intersections
    n_planes = len(scene['planes']['p0'])
    n_spheres = len(scene['spheres']['p0'])
    n_triangles = len(scene['triangles']['p'])
    n_tm = len(scene['triangle_meshes']['faces']) # triangle meshes
    #print('n_tm', n_tm)
    n_intersections = n_planes + 2*n_spheres + n_triangles + n_tm
    t = np.empty((n_rays, n_intersections), dtype=FLOAT_DTYPE)

    i0_plane, i1_plane = (0, n_planes)
    i0_sphere, i1_sphere = (i1_plane, i1_plane+2*n_spheres)
    i0_triangle, i1_triangle = (i1_sphere, i1_sphere+n_triangles)
    i0_tm, i1_tm = (i1_triangle, i1_triangle+n_tm)

    # Plane intersections. Shape = (# of rays, # of planes)
    t[:,i0_plane:i1_plane] = plane_intersection(
        x0, v,
        scene['planes']['p0'],
        scene['planes']['n']
    )

    # Sphere intersections. Shape = (# of rays, # of spheres)
    t[:,i0_sphere:i1_sphere] = sphere_intersection(
        x0, v,
        scene['spheres']['p0'],
        scene['spheres']['r'],
        unroll_multiple_intersections=True
    )

    # Triangle intersections. Shape = (# of rays, # of triangles)
    t[:,i0_triangle:i1_triangle] = triangle_intersection(
        x0, v,
        scene['triangles']['p']
    )

    #n_tm = len(tm['vertices'])
    #for i in range(n_tm):
    #    t_i = triangle_intersection(
    #        x0, v,
    #        expand_triangle_mesh(tm['vertices'][i], tm['faces'][i])
    #    )
    #    print(t_i)

    # Determine which rays intersect which triangle mesh bounding spheres
    t_bounds = sphere_intersection(
        x0, v,
        scene['triangle_meshes']['bounds']['x0'],
        scene['triangle_meshes']['bounds']['r'],
        unroll_multiple_intersections=True
    ) # shape = (ray, 2*mesh)
    #print('t_bounds', t_bounds)
    #print('t_bounds.shape', t_bounds.shape)
    #tm_has_intersections = np.any(t_bounds > 0., axis=1)
    tm = scene['triangle_meshes']
    tm_child_idx = np.empty((n_rays,n_tm), dtype='u2')
    #ray_idx_tm = [] # For each tm, indices of relevant rays
    #close_idx_tm = [] # For each tm, indices of hit triangles
    for i in range(n_tm):
        # Identify rays that travel through bounding volume of mesh
        ray_idx_i = (
            ((t_bounds[:,2*i]>0.) & np.isfinite(t_bounds[:,2*i]))
          | ((t_bounds[:,2*i+1]>0.) & np.isfinite(t_bounds[:,2*i+1]))
        )
        # Find intersections of relevant rays with all triangles in mesh
        p_tm = expand_triangle_mesh(tm['vertices'][i], tm['faces'][i])
        #print('ray_idx_tm_i', ray_idx_i)
        #print(np.count_nonzero(ray_idx_i))
        t_i = triangle_intersection(
            x0[ray_idx_i], v[ray_idx_i],
            expand_triangle_mesh(tm['vertices'][i], tm['faces'][i])
        )
        #print('t_i', t_i)
        #print(f'finite t_i: {np.count_nonzero(np.isfinite(t_i))}')
        #print('t_i.shape', t_i.shape)
        close_idx_i, t_close_i = find_closest_intersections(t_i)
        #print('close_idx_i', close_idx_i)
        #print('t_close_i', t_close_i)
        #print(f'finite t_close_i: {np.count_nonzero(np.isfinite(t_close_i))}')
        t[ray_idx_i,i0_tm+i] = t_close_i
        t[~ray_idx_i,i0_tm+i] = np.inf
        tm_child_idx[ray_idx_i,i] = close_idx_i
        #ray_idx_tm.append(ray_idx_i)
        #close_idx_tm.append(close_idx_i)

    # For each ray, calculate closest intersections
    close_idx, t_close = find_closest_intersections(t)
    #print('close_idx', close_idx)
    #print(np.count_nonzero(close_idx == 1))
    #print('t_close', t_close)

    # Identify rays with an intersection (i.e., t not infinite)
    ray_idx = np.isfinite(t_close)
    ambient_ray_idx = np.where(~ray_idx)[0] # Rays that go to infinity
    ray_idx = np.where(ray_idx)[0]
    obj_idx = close_idx[ray_idx]
    #plane_idx = close_idx[ray_idx]
    #print('ray_idx', ray_idx)
    #print(f'ray_idx.shape = {ray_idx.shape}')
    #print(f'obj_idx.shape = {obj_idx.shape}')

    # Determine what type of object each ray intersects, and determine
    # the index (ID) of that object in its respective object array.
    idx_is_plane = (obj_idx >= i0_plane) & (obj_idx < i1_plane)
    idx_is_sphere = (obj_idx >= i0_sphere) & (obj_idx < i1_sphere)
    idx_is_triangle = (obj_idx >= i0_triangle) & (obj_idx < i1_triangle)
    idx_is_tm = (obj_idx >= i0_tm) #& (obj_idx < i1_tm)
    #print('idx_is_plane', idx_is_plane)
    #print('idx_is_sphere', idx_is_sphere)
    #print('idx_is_triangle', idx_is_triangle)
    #print('idx_is_tm', idx_is_tm)
    #print(f'idx_is_tm: {np.count_nonzero(idx_is_tm)/idx_is_tm.size*100}%')

    plane_id = obj_idx[idx_is_plane]# - i0_plane
    sphere_id = (obj_idx[idx_is_sphere] - i0_sphere) // 2
    triangle_id = obj_idx[idx_is_triangle] - i0_triangle
    tm_id = obj_idx[idx_is_tm] - i0_tm

    ray_idx_plane = ray_idx[idx_is_plane]
    ray_idx_sphere = ray_idx[idx_is_sphere]
    ray_idx_triangle = ray_idx[idx_is_triangle]
    ray_idx_tm = ray_idx[idx_is_tm]

    # Determine sub-indices of intersected triangles in each triangle mesh
    tm_child_id = tm_child_idx[ray_idx_tm, tm_id]
    #print('tm_child_id', tm_child_id)
    #print(np.unique(tm_child_id, return_counts=True))

    #print('plane_id', plane_id)
    #print('sphere_id', sphere_id)
    #print('tm_id', tm_id)

    # Add luminosity from sources that are directly hit
    ray_value[ray_idx_plane] = scene['planes']['color'][plane_id]
    ray_value[ray_idx_sphere] = scene['spheres']['color'][sphere_id]
    ray_value[ray_idx_triangle] = scene['triangles']['color'][triangle_id]
    ray_value[ray_idx_tm] = scene['triangle_meshes']['color'][tm_id]

    # Add ambient light
    ray_value[ambient_ray_idx] = scene['ambient_color'][None,:]

    if (recursion_depth >= recursion_limit) or (len(t_close) == 0):
        if collect_rays:
            ray_props = {
                'x0': [x0],
                'v': [v],
                't': [t_close],
                'n': [np.zeros_like(v)],
                'ray_value': [ray_value]
            }
            return ray_value, ray_props
        return ray_value
    
    x0_child = []
    v_child = []
    child_contrib = []
    child_parent_idx = []

    # Calculate intersection coordinates and incoming direction
    v_i = v[ray_idx]
    x_i = x0[ray_idx] + v_i * t_close[ray_idx][:,None]

    # Determine intersection normals
    n_intersects = ray_idx.shape[0]
    n = np.empty(shape=(n_intersects,n_dim), dtype=FLOAT_DTYPE)
    n[idx_is_plane] = scene['planes']['n'][plane_id]
    n[idx_is_sphere] = sphere_normal(
        x_i[idx_is_sphere],
        scene['spheres']['p0'][sphere_id]
    )
    n[idx_is_triangle] = triangle_normal(
        scene['triangles']['p'][triangle_id]
    )

    idx_is_tm = np.where(idx_is_tm)[0]
    tm = scene['triangle_meshes']
    for i in range(n_tm):
        idx = (tm_id == i)
        tm_child_id_i = tm_child_id[idx]
        verts = scene['triangle_meshes']['vertices'][i]
        faces = scene['triangle_meshes']['faces'][i]
        n[idx_is_tm[idx]] = triangle_normal(
            verts[faces[tm_child_id_i]]
        )

    # Refractive index arrays
    refract = np.empty(shape=(n_intersects,2), dtype=FLOAT_DTYPE)
    refract[idx_is_plane] = scene['planes']['refract'][plane_id]
    refract[idx_is_sphere] = scene['spheres']['refract'][sphere_id]
    refract[idx_is_triangle] = scene['triangles']['refract'][triangle_id]
    refract[idx_is_tm] = scene['triangle_meshes']['refract'][tm_id]

    # Spawn transmitted and reflected rays from refraction
    idx_refract = np.all(np.abs(refract) > 0.999, axis=1)
    v_t,v_r,R = refraction_outgoing(
        v_i[idx_refract],
        n[idx_refract],
        refract[idx_refract,0],
        refract[idx_refract,1]
    )
    for vv,RR in [(v_r,R),(v_t,1-R)]:
        RR.shape = (-1,1)
        RR = np.repeat(RR, n_channels, axis=1)
        x0_child.append(x_i[idx_refract])
        v_child.append(vv)
        child_contrib.append(RR)
        child_parent_idx.append(ray_idx[idx_refract])

    # Empty reflectivity and diffusivity arrays
    reflect = np.empty(shape=(n_intersects,n_channels), dtype=FLOAT_DTYPE)
    diffuse = np.empty(shape=(n_intersects,n_channels), dtype=FLOAT_DTYPE)

    # Plane reflectivity, diffusivity, etc.
    reflect[idx_is_plane] = scene['planes']['reflectivity'][plane_id]
    diffuse[idx_is_plane] = scene['planes']['diffusivity'][plane_id]

    # Sphere reflectivity, diffusivity, etc.
    reflect[idx_is_sphere] = scene['spheres']['reflectivity'][sphere_id]
    diffuse[idx_is_sphere] = scene['spheres']['diffusivity'][sphere_id]

    # Triangle reflectivity, diffusivity, etc.
    reflect[idx_is_triangle] = scene['triangles']['reflectivity'][triangle_id]
    diffuse[idx_is_triangle] = scene['triangles']['diffusivity'][triangle_id]

    # Triangle mesh reflectivity, diffusivity, etc.
    reflect[idx_is_tm] = scene['triangle_meshes']['reflectivity'][tm_id]
    diffuse[idx_is_tm] = scene['triangle_meshes']['diffusivity'][tm_id]

    # Spawn mirror reflection at each intersection
    x0_child.append(x_i)
    v_child.append(mirror_reflection_outgoing(v_i, n))
    child_contrib.append(reflect)
    child_parent_idx.append(ray_idx)

    # Spawn diffuse reflection at each intersection
    for k in range(n_diffuse):
        x0_child.append(x_i)
        vo,cos_phi = diffuse_reflection_outgoing(v_i, n, rng)
        v_child.append(vo)
        child_contrib.append(diffuse*cos_phi[:,None]/n_diffuse)
        child_parent_idx.append(ray_idx)

    # Combine all types of child rays into one array
    x0_child = np.concatenate(x0_child, axis=0)
    v_child = np.concatenate(v_child, axis=0)
    child_contrib = np.concatenate(child_contrib, axis=0)
    child_parent_idx = np.concatenate(child_parent_idx, axis=0)

    # Filter out rays with zero contribution
    idx_contrib = np.any(child_contrib>1e-7, axis=1)
    #print(
    #    f'{np.count_nonzero(idx_contrib)} of {idx_contrib.size} '
    #    'rays contribute.'
    #)
    x0_child = x0_child[idx_contrib]
    v_child = v_child[idx_contrib]
    child_contrib = child_contrib[idx_contrib]
    child_parent_idx = child_parent_idx[idx_contrib]

    #print(f'recursion depth: {recursion_depth}')
    #print('x0:\n', x0_child)
    #print('v:\n', v_child)
    #print('')

    # Recursion: Add in values of spawned rays
    ret = render_rays_recursive(
        recursion_limit,
        x0_child, v_child,
        scene,
        n_diffuse=n_diffuse,
        collect_rays=collect_rays,
        recursion_depth=recursion_depth+1,
        rng=rng
    )
    if collect_rays:
        ray_value_ret, ray_props_ret = ret
    else:
        ray_value_ret = ret

    #print(f'ray_value.shape = {ray_value.shape}')
    #print(f'child_parent_idx.shape = {child_parent_idx.shape}')
    #print(f'child_contrib.shape = {child_contrib.shape}')
    #print(f'ray_value_ret.shape = {ray_value_ret.shape}')
    np.add.at(ray_value, child_parent_idx, child_contrib*ray_value_ret)

    if collect_rays:
        ray_props = {
            'x0': [x0],
            'v': [v],
            't': [t_close],
            'n': [n],
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


def gnomonic_projection(fov, shape,
                        flatten=False,
                        antialias=False,
                        rng=None):
    r = 0.5 * shape[0] / np.tan(np.radians(fov))
    screen_coords = np.indices(shape).astype(FLOAT_DTYPE)
    #screen_coords = np.flip(screen_coords, axis=0)
    for i,s in enumerate(shape):
        screen_coords[i] -= 0.5 * (s-1)

    if antialias:
        if rng is None:
            rng = np.random.default_rng()
        screen_coords += rng.uniform(-0.5, 0.5, size=screen_coords.shape)

    screen_coords = np.concatenate(
        [screen_coords,np.full((1,)+shape,r)],
        axis=0
    ).astype(FLOAT_DTYPE)
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


def gen_camera_rays(camera, flatten=False, antialias=False, rng=None):
    rays = {}
    rays['v'] = gnomonic_projection(
        camera['fov'],
        camera['shape'],
        flatten=flatten,
        antialias=antialias,
        rng=rng
    )
    rays['x0'] = np.tile(
        camera['x0'],
        (rays['v'].shape[0], 1)
    ).astype(FLOAT_DTYPE)
    return rays


def main():
    #x = np.arange(12)
    #x.shape = (4,3)
    #print(x)
    #rotate_vectors(x, 0, 1, np.pi/4.)
    #return 0

    rng = np.random.default_rng(5)

    # Generate random scene
    #n_dim = 2
    #camera_shape = (1,)
    #n_planes = 4
    #n_spheres = 3
    #camera, scene = gen_rand_scene(
    #    n_dim, camera_shape,
    #    n_planes, n_spheres,
    #    rng=rng
    #)
    #print(scene)

    scene_fname = 'triangle_mesh.json'
    #scene_fname = 'plane_with_sphere.json'
    #scene_fname = 'diffuse_box_with_light.json'
    #scene_fname = 'test_scene_2d.json'
    #scene_fname = 'test_refraction_2d_simple.json'
    camera, scene = load_scene(scene_fname)
    n_dim = scene['n_dim']
    camera_shape = camera['shape']
    print(scene)

    # Plot scene
    if n_dim == 2:
        camera_rays = gen_camera_rays(camera, flatten=True)
        fig,ax = plot_scene(camera_rays, scene, recursion_depth=2, rng=rng)
        plt_fname_base = os.path.splitext(scene_fname)[0]
        fig.savefig(
            f'plots/{plt_fname_base}_ray_diagram.svg',
            transparent=False
        )
        xlim = ax.get_xlim()
        xlim = (3*xlim[0], 3*xlim[1])
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        fig.savefig(
            f'plots/{plt_fname_base}_ray_diagram_zoomout.svg',
            transparent=False
        )
        #plt.show()

    # Render scene
    if n_dim != 3:
        return 0

    from tqdm import tqdm
    n_frames = 60
    n_samples = 64
    gamma = 0.20
    #scene_name = 'bobbing_spheres'#'diffuse_box_with_light'
    scene_name = 'triangle_mesh_spheres'

    #spheres_p0 = scene['spheres']['p0'].copy()
    verts = scene['triangle_meshes']['vertices'][0].copy()
    rotate_vectors(
        verts,
        2, 1,
        -0.1*np.pi
    )

    for max_depth in range(5,6):
        print(f'Rendering scene at max depth {max_depth} ...')
        n_pix = np.prod(camera_shape)
        pixel_value_max = None

        for frame in tqdm(range(n_frames)):
            phi = 2*np.pi * frame/n_frames
            scene['triangle_meshes']['vertices'][0] = verts.copy()
            rotate_vectors(
                scene['triangle_meshes']['vertices'][0],
                2, 0,
                -phi
            )

            pixel_color = np.zeros(
                (n_pix, scene['n_channels']),
                dtype=FLOAT_DTYPE
            )
            for k in tqdm(range(n_samples)):
                camera_rays = gen_camera_rays(
                    camera,
                    flatten=True,
                    antialias=True,
                    rng=rng
                )
                #rotate_vectors(camera_rays['v'], 2, 0, -0.10*np.pi*np.sin(phi))
                #rotate_vectors(camera_rays['v'], 2, 0, -phi)
                #camera_rays['x0'][:,0] += 0.3 * np.sin(phi)

                #dp0 = np.array([
                #    [0.1*np.cos(phi), 0.3*np.sin(phi), 0.],
                #    [0., 0., 0.],
                #    [0., 0.5*np.cos(phi), 0.],
                #    [0., 0., 0.],
                #    [0., 0., 0.]
                #])
                #scene['spheres']['p0'] = spheres_p0+dp0

                pixel_color += render_rays_recursive(
                    max_depth,
                    camera_rays['x0'],
                    camera_rays['v'],
                    scene,
                    n_diffuse=3,
                    collect_rays=False,
                    rng=rng
                )
            pixel_intensity = np.linalg.norm(pixel_color, axis=1)
            #print('pixel_intensity', pixel_intensity)
            #print('pixel_color', pixel_color)
            pixel_color *= (pixel_intensity**(gamma-1.))[:,None]
            pixel_color[~np.isfinite(pixel_color)] = 0
            #print('pixel_color', pixel_color)
            #print(f'pixel_color.shape = {pixel_color.shape}')
            #print('max(pixel_color):', np.max(pixel_color))
            if pixel_value_max is None:
                pixel_value_max = np.percentile(pixel_color, 99.5)
            #print(np.percentile(pixel_color, [90., 95., 99., 99.5]))
            #pixel_color /= np.max(pixel_color)
            pixel_color /= pixel_value_max
            if n_dim == 2:
                pixel_color.shape = (1,)+pixel_color.shape
            elif n_dim == 3:
                pixel_color.shape = camera_shape + (3,)
                pixel_color = np.swapaxes(pixel_color, 0, 1)
            pixel_color = np.clip(255*pixel_color,0.,255.).astype('u1')
            #print(pixel_color.shape)
            im = Image.fromarray(pixel_color[::-1,:,:], mode='RGB')
            im.save(
                f'frames/{scene_name}'
                f'_maxdepth{max_depth}'
                f'_frame{frame:05d}'
                '.png'
            )

            #print(v_rot)
            #print(pixel_color)

    return 0

if __name__ == '__main__':
    main()

