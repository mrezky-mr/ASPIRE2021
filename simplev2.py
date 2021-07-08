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

#integrator options
sim.integrator = "ias15"
sim.gravity = "compensated"
sim.softening = 6.68458712e-9   #based on sigma_{c,num}, in au
sim.dt = 0.025*2.*np.pi
sim.testparticle_type = 1
sim.ri_ias15.epsilon = 1e-6

#collision and boundary options
sim.collision = "direct"
sim.coefficient_of_resitution = 0.1
sim.collision_resolve = "hardsphere"
sim.collision_resolve_keep_sorted = 1
sim.track_energy_offset = 1

def rand_uniform(minimum, maximum):
    return np.random.uniform()*(maximum-minimum)+minimum

N_pl = 1000                         # Number of planetesimals
Mtot_disk = 4.188e18                # Total mass of planetesimal disk, in kg
m_pl = Mtot_disk / float(N_pl)      # Mass of each planetesimal, in kg
r_pl = 1.19253034e-7                # Radius of each planetesimal, in au
r_h = 4.44372861e-4                 # Hill's radius, in au
v_dp = 2.10949526e-5                # Maximum velocity dispersion, in au/yr
orb = 44                            # Number of orbit for integration

np.random.seed(50) #by setting a seed we will reproduce the same simulation every time
while sim.N < N_pl:
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
    p = rebound.Particle(simulation=sim, m=m_pl, r=r_pl, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    sim.add(p)

sim.move_to_com()
E0 = sim.calculate_energy()

coords = np.zeros((3,sim.N))
for i in range(sim.N):
    coords[0][i], coords[1][i], coords[2][i] = sim.particles[i].x, sim.particles[i].y, sim.particles[i].z

fig1, ax1 = plt.subplots()
ax1.axis('equal')
ax1.scatter(coords[0],coords[1], s=2)
plt.xlabel('X')
plt.ylabel('Y')
fig1.tight_layout()
plt.savefig("initial_2d.png")
plt.close(fig1)

fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.scatter3D(coords[0],coords[1],coords[2], s=2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
fig2.tight_layout()
plt.savefig("initial_3d.png")
plt.close(fig2)

ts = int(orb*2*np.pi)
times = np.linspace(0.,ts,ts)
totalN = np.zeros(len(times))
errors = np.zeros(len(times))
for i,t in enumerate(times):
    sim.integrate(t)
    totalN[i] = sim.N
    errors[i] = abs((sim.calculate_energy() - E0)/E0)
    if t % 10 < 1:
        coords = np.zeros((5,sim.N))
        for i in range(sim.N):
            coords[0][i], coords[1][i], coords[2][i] = sim.particles[i].x, sim.particles[i].y, sim.particles[i].z
            coords[3][i], coords[4][i] = sim.particles[i].vx, sim.particles[i].vy
        fig3, ax3 = plt.subplots()
        plt.title("t = {a}".format(a=int(t)), loc='right')
        ax3.axis('equal')
        ax3.scatter(coords[0],coords[1], s=2)
        ax3.scatter(sim.particles[0].x,sim.particles[0].y, s=2)
        ax3.quiver(coords[0],coords[1],coords[3],coords[4])
        plt.xlabel('X')
        plt.ylabel('Y')
        fig3.tight_layout()
        plt.savefig("t_{a}_2d.png".format(a=int(t)))
        plt.close(fig3)
        
        fig4 = plt.figure()
        ax4 = plt.axes(projection='3d')
        ax4.scatter3D(coords[0],coords[1],coords[2], s=2)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        fig4.tight_layout()
        plt.title("t = {a}".format(a=int(t)), loc='right')
        plt.savefig("t_{a}_3d.png".format(a=int(t)))
        plt.close(fig4)
    print(t)

fig5 = plt.figure(figsize=(10,7))
axa = plt.subplot(211)
plt.yscale("log")
plt.plot(times/(2.*np.pi), errors)
axa.set_ylabel("Relative energy error")
axb = plt.subplot(212)
axb.set_ylabel("Lost/merged particles")
axb.set_xlabel("Time [orbits]")
plt.plot(times/(2.*np.pi), -(totalN-N_pl-2))
plt.savefig("stats.png")
plt.close(fig5)

p_mass = []
p_rad = []
p_v = []

coords = np.zeros((3,sim.N))
for i in range(sim.N):
    coords[0][i], coords[1][i], coords[2][i] = sim.particles[i].x, sim.particles[i].y, sim.particles[i].z
    p_mass.append(sim.particles[i].m)
    p_rad.append(sim.particles[i].r)
    x = np.array([sim.particles[i].vx, sim.particles[i].vy])
    p_v.append(np.linalg.norm(x)*149597870700)
    
dat = np.vstack((p_mass, p_rad, p_v))
np.savetxt("final_pdata.txt", dat)
    
fig6, ax6 = plt.subplots() 
ax6.axis('equal')
ax6.scatter(coords[0],coords[1], s=2)
plt.xlabel('X')
plt.ylabel('Y')
fig6.tight_layout()
plt.savefig("final_2d.png")
plt.close(fig6)

fig7 = plt.figure()
ax7 = plt.axes(projection='3d')
ax7.scatter3D(coords[0],coords[1],coords[2], s=2)
ax7.set_xlabel('X')
ax7.set_ylabel('Y')
ax7.set_zlabel('Z')
fig7 .tight_layout()
plt.savefig("final_3d.png")
plt.close(fig7)

print("minimum velocity (m/s) = {a}".format(a=min(p_v)))
print("maximum velocity (m/s) = {a}".format(a=max(p_v)))
print("minimum radius (au) = {a}".format(a=min(p_rad)))
print("maximum radius (au) = {a}".format(a=max(p_rad)))
print("minimum mass (kg) = {a}".format(a=min(p_mass)))
print("maximum mass (kg) = {a}".format(a=max(p_mass)))