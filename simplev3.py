#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 00:42:22 2021

@author: mrezky
"""

import rebound
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

sim = rebound.Simulation()
sim.units = ('yr','AU','kg')

plt.ioff()
mpl.use('Agg')

au = 1.495978707e11             # astronomical units, in m
y = 365.25*24*3600              # year-to-second conversion

#sphere properties
def sphr(a, M, rho, R):
    m = rho * (4/3) * np.pi * R**3              # total mass of the sphere, in kg
    r = a * (m/(3*M))**(1/3)                    # Hill's radius, in au
    P = 2*np.pi*np.sqrt(a**3 / (sim.G * M))     # orbital timescale, in yr
    rho_sp = 3 * m / (4 * np.pi * r**3)         # sphere density, in kg au-3
    tff = np.sqrt(3*np.pi/(32*sim.G*rho_sp))    # free-fall time, in yr
    o = 2*tff / P                               # number of orbits for simulation
    return m, r, o, tff

#simulated particle properties
def parpr(m, r, s, Nn):
    rh_1 = 3*m / (4*np.pi*r**3)     # density of the sphere dedicated to PX, in kg m-3
    rh_s = 1e-6                     # density of a particle in real system, in kg m-3
    N = rh_1/rh_s                   # number of particles in real system
    sn = s * N / Nn                 # coll. xsection in sim, in m
    an = np.sqrt(sn/np.pi)          # radius of a particle, in m
    mp = m/Nn                       # mass of a particle in kg
    #mp = rh_s*(4/3)*np.pi*(an)**3  
    return an, mp

#uniformed random sequencer
def rand_uniform(minimum, maximum):
    return np.random.uniform()*(maximum-minimum)+minimum

#generate particle for each species
def particles(N, m, rp):
    n = 0
    while n < N:
        u = rand_uniform(0, 1)
        ct = rand_uniform(-1, 1)
        r = r_h * u**(1/3)
        theta = np.arccos(ct)
        phi = rand_uniform(0, 2*np.pi)
        x = r*np.cos(phi)*np.sin(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(theta)
        vx = rand_uniform(-v_dp, v_dp)
        vy = rand_uniform(-v_dp, v_dp)
        vz = rand_uniform(-v_dp, v_dp)
        p = rebound.Particle(simulation=sim, m=m, r=rp, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
        sim.add(p)
        n += 1
    return sim

#integrator options
sim.integrator = "ias15"
sim.gravity = "compensated"
sim.dt = 0.002*2.*np.pi
sim.testparticle_type = 1
sim.ri_ias15.epsilon = 1e-8

#collision and boundary options
sim.collision = "direct"
sim.coefficient_of_resitution = 0.1
sim.collision_resolve = "hardsphere"
sim.collision_resolve_keep_sorted = 1
sim.track_energy_offset = 1

#sphere parameters
M_star = 1.989e30                   # mass of the central star, in kg
r_p = 100000                        # radius of supposed planet, in m
rho_p = 1000                        # density of supposed planet, in kg m-3
a = 5                               # sphere distance from the star, in au
m_p, r_h, orb, tff = sphr(a, M_star, rho_p, r_p)

#scaled-particle parameters, for species (P) 1 and 2
N_pl1 = 1000                 # number of particles in sim (P1)
N_pl2 = 1000                 # number of particles in sim (P2)
v_dp = 0.1 * y / au          # Maximum velocity dispersion in m/s, converted to au/yr
s_cr1 = 1                    # real collisional xsection of P1, in m
s_cr2 = 0.01                 # real collisional xsection of P2, in m
rat1 = 0.3
rat2 = 1 - rat1
m_pl1 = rat1 * m_p           # total mass of P1 particles, in kg
m_pl2 = rat2 * m_p           # total mass of P2 particles, in kg

a1, mp1 = parpr(m_pl1, r_p, s_cr1, N_pl1)
a2, mp2 = parpr(m_pl2, r_p, s_cr2, N_pl2)

sim.softening = min(a1,a2)/au  # based on a_{c,num}, in au

np.random.seed(70)
particles(N_pl1, mp1, a1/au)
Na = sim.N
particles(N_pl2, mp2, a2/au)

sim.move_to_com()
E0 = sim.calculate_energy()
N0 = sim.N
print("Total particles: ", N0)
print('-----')

ts = orb*2*np.pi
times = np.linspace(0.,ts,50)
Coll = np.zeros(N0)
Ncol = np.zeros(len(times))
errors = np.zeros(len(times))
print('Numerical integration is on progress')
for i,t in enumerate(times):
    sim.integrate(t)
    k = 0
    ncol = 0
    for k in range(sim.N):
        if Coll[k] < sim.particles[k].lastcollision:
            ncol += 1
            Coll[k] = sim.particles[k].lastcollision
    errors[i] = abs((sim.calculate_energy() - E0)/E0)
    if i % 5 == 0:
        coords = np.zeros((5,sim.N))
        for j in range(sim.N):
            coords[0][j], coords[1][j], coords[2][j] = sim.particles[j].x, sim.particles[j].y, sim.particles[j].z
            coords[3][j], coords[4][j] = sim.particles[j].vx * au/y, sim.particles[j].vy * au/y
        fig3, ax3 = plt.subplots()
        plt.title("t = {a}".format(a=t))
        ax3.axis('equal')
        ax3.scatter(coords[0][:Na],coords[1][:Na], s=4, label=r"$\sigma_c = {n} \ m$".format(n=s_cr1))
        ax3.scatter(coords[0][Na:],coords[1][Na:], s=4, label=r"$\sigma_c = {n} \ m$".format(n=s_cr2))
        ax3.quiver(coords[0],coords[1],coords[3],coords[4])
        ax3.xaxis.set_major_locator(plt.MultipleLocator(max(coords[0])*1.5))
        ax3.yaxis.set_major_locator(plt.MultipleLocator(max(coords[1])*1.5))
        plt.xlabel('X (au)')
        plt.ylabel('Y (au)')
        plt.legend()
        fig3.tight_layout()
        plt.savefig("2d_{a}.png".format(a=i))
        plt.close(fig3)
        
        fig4 = plt.figure(figsize=(10,10))
        ax4 = plt.axes(projection='3d')
        ax4.scatter3D(coords[0][:Na],coords[1][:Na],coords[2][:Na], s=10, label=r"$\sigma_c = {n} m$".format(n=s_cr1))
        ax4.scatter3D(coords[0][Na:],coords[1][Na:],coords[2][Na:], s=10, label=r"$\sigma_c = {n} m$".format(n=s_cr2))
        ax4.set_xlabel('X (au)')
        ax4.set_ylabel('Y (au)')
        ax4.set_zlabel('Z (au)')
        ax4.set_xlim(-r_h, r_h)
        ax4.set_ylim(-r_h, r_h)
        ax4.set_zlim(-r_h, r_h)
        plt.title("t = {a}".format(a=t))
        plt.legend()
        fig4.tight_layout()
        plt.savefig("3d_{a}.png".format(a=i))
        plt.close(fig4)
    sim.simulationarchive_snapshot('simplev3.bin')
    Ncol[i] = ncol
    print(t)

fig5 = plt.figure(figsize=(10,7))
axa = plt.subplot(211)
plt.yscale("log")
plt.plot(times/(np.pi*orb), errors)
axa.set_ylabel("Relative energy error")
axb = plt.subplot(212)
axb.set_ylabel(r"$N_{c,num} / N_0$")
axb.set_xlabel(r"Time [$t_{ff}$]")
plt.plot(times/(np.pi*orb), Ncol/N0)
plt.savefig("stats.png")
plt.close(fig5)

print("Total mass of the sphere: {a} kg".format(a=m_p))
print("Free-fall time: {a} seconds".format(a=tff*y))
print("-----")
print("Mass ratio for P1: {a} of total mass".format(a=rat1))
print("Radius of a particle (P1): {a} km".format(a=a1/np.sqrt(N_pl1)))
print("Mass of a particle (P1): {a} kg".format(a=mp1))
print("-----")
print("Mass ratio for P2: {a} of total mass".format(a=rat2))
print("Radius of a particle (P2): {a} km".format(a=a2/np.sqrt(N_pl2)))
print("Mass of a particle (P2): {a} kg".format(a=mp2))