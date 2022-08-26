#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def plane_intersection(x0, v, p0, n):
    """
    Compute the intersections of a set of rays with a set of planes.

    Inputs:
        x0 (np.ndarray): Coords of ray origin. Shape = (# of rays, # of dims).
        v (np.ndarray): Direction of ray. Shape = (# of rays, # of dims).
        p0 (np.ndarray): Point on plane. Shape = (# of planes, # of dims).
        n (np.ndarray): Normal to plane. Shape = (# of planes, # of dims).

    Outputs:
        t (np.ndarray): Intersection dists. Shape = (# of rays, # of planes).
    """
    num = np.sum((p0[None,:,:]-x0[:,None,:])*n[None,:,:], axis=2)
    denom = np.sum(v[:,None,:]*n[None,:,:], axis=2)
    return num / denom


def find_closest_intersections(t):
    t[t<0] = np.inf
    idx = np.argmin(t, axis=1)
    return idx, t[np.arange(len(idx)),idx]


def plot_scene(x0, v, p0, n, c):
    n_dim = x0.shape[1]

    # Calculate intersections
    t = plane_intersection(x0, v, p0, n)
    close_idx, t_close = find_closest_intersections(t.copy())

    for tt,ii in zip(t_close, close_idx):
        print(tt, ii)

    # Get ray colors
    ray_color = render_rays(x0, v, p0, n, c)
    print('ray_color:\n', ray_color)

    # Figure
    fig,ax = plt.subplots(
        1,1, figsize=(6,6),
        subplot_kw=dict(aspect='equal')
    )

    # Draw planes
    for pp,nn,cc in zip(p0,n,c):
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
            alpha=0.5,
            lw=3
        )

    # Draw rays
    ax.scatter(x0[:,0], x0[:,1], c='blue')
    for xx,vv in zip(x0,v):
        ax.arrow(
            xx[0], xx[1],
            0.5*vv[0], 0.5*vv[1],
            color='blue',
            head_width=0.1
        )
        #ax.plot(
        #    [xx[0]-30*vv[0], xx[0]+30*vv[0]],
        #    [xx[1]-30*vv[1], xx[1]+30*vv[1]],
        #    color='blue',
        #    alpha=0.1
        #)

    # Draw intersections
    #xp = x0[:,None,:] + v[:,None,:]*t[:,:,None]
    #xp = np.reshape(xp, (-1,n_dim))
    #ax.scatter(xp[:,0], xp[:,1], c='k', alpha=0.2)

    # Draw closest intersections
    idx = np.isfinite(t_close)
    xp = x0[idx,:] + v[idx,:]*t_close[idx,None]
    xp = np.reshape(xp, (-1,n_dim))
    for xo,xx,cc in zip(x0[idx], xp, ray_color[idx]):
        ax.scatter([xx[0]], [xx[1]], color=cc)
        ax.plot([xo[0],xx[0]],[xo[1],xx[1]], color=cc, alpha=0.2)

    # Draw rays that go to infinity
    for xo,vv in zip(x0[~idx], v[~idx]):
        ax.plot(
            [xo[0], xo[0]+30*vv[0]],
            [xo[1], xo[1]+30*vv[1]],
            color='k',
            alpha=0.05
        )


    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    return fig


def gen_rand_scene(n_dim, camera_shape, n_planes, fov=75., rng=None):
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
    n = rng.normal(size=(n_planes, n_dim))
    n /= np.linalg.norm(n, axis=1)[:,None]

    # Plane colors
    c = rng.uniform(size=(n_planes, 3))

    return x0, v, p0, n, c


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
    ray_value = np.zeros((n_rays, n_channels), dtype=c.dtype)
    ray_value[ray_idx] = c[plane_idx]

    return ray_value


def gnomonic_projection(fov, shape, flatten=False, dtype='f8'):
    r = 0.5 * shape[0] / np.tan(np.radians(fov))
    screen_coords = np.indices(shape).astype(dtype)
    #screen_coords = np.flip(screen_coords, axis=0)
    for i,s in enumerate(shape):
        screen_coords[i] -= 0.5 * (s-1)
    screen_coords = np.concatenate(
        [screen_coords,np.full((1,)+shape,r,dtype=dtype)],
        axis=0
    )
    screen_coords /= np.linalg.norm(screen_coords, axis=0)[None]
    print(screen_coords.shape)
    screen_coords = np.moveaxis(screen_coords, 0, -1)
    if flatten:
        print(screen_coords.shape)
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

    rng = np.random.default_rng(4)

    # Generate random scene
    n_dim = 2
    camera_shape = (20,)
    n_planes = 4
    x0, v, p0, n, c = gen_rand_scene(n_dim, camera_shape, n_planes, rng=rng)

    # Plot scene
    fig = plot_scene(x0, v, p0, n, c)
    plt.show()

    # Render scene
    pixel_color = render_rays(x0, v, p0, n, c)
    if n_dim == 2:
        pixel_color.shape = (1,)+pixel_color.shape
    elif n_dim == 3:
        pixel_color.shape = camera_shape + (3,)
        pixel_color = np.swapaxes(pixel_color, 0, 1)
    pixel_color = np.clip(255*pixel_color,0.,255.).astype('u1')
    print(pixel_color.shape)
    im = Image.fromarray(pixel_color, mode='RGB')
    im.save('rendered_scene.png')
    print(pixel_color)

    return 0

if __name__ == '__main__':
    main()

