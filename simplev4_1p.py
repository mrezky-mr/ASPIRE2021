#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 00:42:22 2021

@author: mrezky
"""

import rebound
import numpy as np
import matplotlib as mpl
from time import time, ctime
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#Indicates the starting time of this code run
start = ctime(time())
print(start)

sim = rebound.Simulation()
sim.units = ('yr','AU','kg')

plt.ioff()
mpl.use('Agg')

au = 1.495978707e11             # astronomical units, in m
y = 365.25*24*3600              # year-to-second conversion

#sphere/core properties
def sphr(a, M, rho, R):
    m = rho * (4/3) * np.pi * R**3                  # total mass of the core, in kg
    r = a * (m/(3*M))**(1/3)                        # Hill's radius, in au
    P = 2*np.pi*np.sqrt(a**3 / (sim.G * M))         # orbital timescale, in yr
    rho_c = 3 * m / (4 * np.pi * r**3 * au**3)      # core density, in kg au-3 
    tfs = np.sqrt(3*np.pi/(32*6.674e-11*rho_c))     # free-fall time, in s
    tfy = tfs/y                                     # free-fall time, in yr
    return m, r, tfy, P

#simulated particle properties
def parpr(m, d, s, Nn):
    mpr = (4/3)*np.pi*(d**3)*rho_p  # mass of each particles, in kg
    N = m/mpr                       # the amount of particles in real system
    sn = s * N / Nn                 # coll. xsection in sim, in m2
    an = np.sqrt(sn/np.pi)          # radius of a particle, in m
    mp = m/Nn                       # mass of a particle in kg
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

def cor_bridges(r, v):
	eps = cor						
	return eps

# STANDARD MODEL INPUT ###################################################
#integrator options
sim.integrator = "ias15"
sim.gravity = "compensated"
sim.dt = 0.0002*2.*np.pi
sim.testparticle_type = 1
sim.ri_ias15.epsilon = 1e-6
sof_mp = 1                      #softening param. multiplier, if necessary

#collision and boundary options
sim.collision = "direct"
cor = 0.1
sim.coefficient_of_restitution = cor_bridges
sim.collision_resolve = "hardsphere"
sim.track_energy_offset = 1

#core properties
M_star = 1.989e30   # mass of the central star, in kg
r_p = 100000        # radius of supposed planet, in m
rho_p = 1000        # density of supposed planet, in kg m-3
a = 5               # sphere distance from the star, in au

#scaled-particle parameters, for 1 species of particle
N_pl = 1000         # number of particles in sim
v = 0.1             # maximum velocity dispersion in m/s, 0 = error
af1 = 1             # radius of a particle in real system, in m

##############################################################################

#calculating derived parameters from the inputs
v_dp = v * y / au                               # v converted to au/yr
m_p, r_h, tff, P = sphr(a, M_star, rho_p, r_p)  # finding core physical and dynamical properties
s_cr1 = np.pi*(af1**2)                          # particle's collisional xsection in real system, in m2
a1, mp1 = parpr(m_p, af1, s_cr1, N_pl)          # finding the mass and radius of a particle in the simulation
f = (N_pl* a1**3)/(r_h**3)                      # filling factor
sim.softening = sof_mp*a1/au                    # based on the highest a_{c,num}, in au

#particle generating process
np.random.seed(70)
particles(N_pl, mp1, a1/au)

#calculate the initial number of particles, energy, and angular momentum of the system
sim.move_to_com()
smx, smy, smz = sim.calculate_angular_momentum()
am0 = np.sqrt(smx**2 + smy**2 + smz**2)
E0 = sim.calculate_energy()
N0 = sim.N
print("Total particles: ", N0)
print('-----')

#initiate the timestep and empty arrays for data collection
times = np.linspace(0.,1.3*tff,50)
Ncol = np.zeros(len(times))
errors = np.zeros(len(times))
am = np.zeros(len(times))

#starting the integration and data collection for each timestep
print('NUMERICAL INTEGRATION IS IN PROGRESS')
for i,t in enumerate(times):
    sim.integrate(t)
    errors[i] = (sim.calculate_energy() - E0)/E0
    amx, amy, amz = sim.calculate_angular_momentum()
    am[i] = (np.sqrt(amx**2 + amy**2 + amz**2) - am0)/am0
    Ncol[i] = sim.collisions_Nlog/N0
    if i % 3 == 0:
        coords = np.zeros((5,sim.N))
           
        #plotting the position and velocity vector of the particles (2D)
        for j in range(sim.N):
            coords[0][j] = sim.particles[j].x
            coords[1][j] = sim.particles[j].y
            coords[2][j] = sim.particles[j].z
            coords[3][j] = sim.particles[j].vx
            coords[4][j] = sim.particles[j].vy
        fig3, ax3 = plt.subplots()
        plt.title(r"$t = {a} \times tff$".format(a=round(t/tff,4)))
        ax3.quiver(coords[0],coords[1],coords[3],coords[4])
        ax3.scatter(coords[0],coords[1], s=4, label=r"$\sigma_c = {n} \ \text{m}^2$".format(n=round(s_cr1,2)))
        ax3.set_xlim(-r_h, r_h)
        ax3.set_ylim(-r_h, r_h)
        ax3.axis('equal')
        plt.xlabel('X (au)')
        plt.ylabel('Y (au)')
        plt.legend(loc='upper right')
        fig3.tight_layout()
        plt.savefig("2d_{a}.png".format(a=i))
        plt.close(fig3)
        
        #plotting the position of the particles (3D)
        fig4 = plt.figure(figsize=(10,10))
        ax4 = plt.axes(projection='3d')
        ax4.scatter3D(coords[0],coords[1],coords[2], s=10, label=r"$\sigma_c = {n} \ \text{m}^2$".format(n=round(s_cr1,2)))
        ax4.set_xlabel('X (au)')
        ax4.set_ylabel('Y (au)')
        ax4.set_zlabel('Z (au)')
        ax4.set_xlim(-r_h, r_h)
        ax4.set_ylim(-r_h, r_h)
        ax4.set_zlim(-r_h, r_h)
        plt.title(r"$t = {a} \times tff$".format(a=round(t/tff,4)))
        plt.legend()
        fig4.tight_layout()
        plt.savefig("3d_{a}.png".format(a=i))
        plt.close(fig4)
    sim.simulationarchive_snapshot('simplev3.bin')
    print('-----')
    print("coll:", sim.collisions_Nlog)
    print("energy:", errors[i])
    print("ang. momentum:", am[i])
    print("time:", t/tff, "of tff")
    
#plotting the energy offset and collision number of the simulation
fig5 = plt.figure(figsize=(7,10))
axa = plt.subplot(311)
plt.plot(times/tff, errors)
plt.yscale("log")
axa.set_ylabel(r"Rel. $E$ error")
axc = plt.subplot(312)
plt.plot(times/tff, am)
axc.set_ylabel(r"Rel. $l$ error")
axb = plt.subplot(313)
plt.plot(times/tff, Ncol)
plt.yscale("log")
axb.set_ylabel(r"$N_{c,\text{num}} / N_0$")
axb.set_xlabel(r"Time [$t_{\text{ff}}$]")
plt.savefig("stats.png")
plt.close(fig5)

coords = np.zeros((5,sim.N))
for j in range(sim.N):
    coords[0][j] = sim.particles[j].x
    coords[1][j] = sim.particles[j].y
    coords[2][j] = sim.particles[j].z
    coords[3][j] = sim.particles[j].vx
    coords[4][j] = sim.particles[j].vy

#plotting the final position and velocity vector of the particles (2D)
fig6, ax6 = plt.subplots()
ax6.quiver(coords[0],coords[1],coords[3],coords[4])
ax6.scatter(coords[0],coords[1], s=4, label=r"$\sigma_c = {n} \ \text{m}^2$".format(n=round(s_cr1,2)))
ax6.set_xlim(-r_h, r_h)
ax6.set_ylim(-r_h, r_h)
plt.xlabel('X (au)')
plt.ylabel('Y (au)')
ax6.axis('equal')
plt.legend(loc='upper right')
plt.title(r"$t = {a} \times tff$".format(a=round(t/tff,4)))
fig6.tight_layout()
plt.savefig("2d_end.png")
plt.close(fig6)

#plotting the final position of the particles (3D)
fig7 = plt.figure(figsize=(10,10))
ax7 = plt.axes(projection='3d')
ax7.scatter3D(coords[0],coords[1],coords[2], s=10, label=r"$\sigma_c = {n} \ \text{m}^2$".format(n=round(s_cr1,2)))
ax7.set_xlabel('X (au)')
ax7.set_ylabel('Y (au)')
ax7.set_zlabel('Z (au)')
ax7.set_xlim(-r_h, r_h)
ax7.set_ylim(-r_h, r_h)
ax7.set_zlim(-r_h, r_h)
plt.legend()
plt.title(r"$t = {a} \times tff$".format(a=round(t/tff,4)))
fig7.tight_layout()
plt.savefig("3d_end.png")
plt.close(fig7)

print('-----')
print('NUMERICAL INTEGRATION IS FINISHED')
print("-----")

#Indicates the finish time of this code run
finish = ctime(time())
print(finish)

file = open("summary.txt","w")
file.write("SUMMARY OF THE SIMULATION")
file.write("\n")
file.write("The program started on {a} \n".format(a=start))
file.write("and ended on {a} \n".format(a=finish))
file.write("---------- \n")
file.write("INPUT PARAMETERS \n")
file.write("The mass of the central star: {a} kg \n".format(a=M_star))
file.write("The radius of the core: {a} m \n".format(a=r_p))
file.write("The density of the core: {a} kg m-3 \n".format(a=rho_p))
file.write("Distance between the core and its central star: {b} au \n".format(b=a))
file.write("The maximum velocity dispersion of the particles: {a} m/s \n".format(a=v))
file.write("Radius of the physical particles: {a} m \n".format(a=af1))
file.write("The amount of particles in the simulation: {a} \n".format(a=N_pl))
file.write("The coefficient of restitution: {a} \n".format(a=cor))
file.write("Tolerance: {a} \n".format(a=sim.ri_ias15.epsilon))
file.write("---------- \n")
file.write("CALCULATED PARAMETERS \n")
file.write("Total mass of the core: {a} kg \n".format(a=m_p))
file.write("Softening parameters: {a} m \n".format(a=sim.softening*au))
file.write("Filling factor: {a}\n".format(a=f))
file.write("Free-fall time: {a} years \n".format(a=tff))
file.write("Orbital timescale: {a} years \n".format(a=P))
file.write("This numerical integration was done for {a} of its orbital timescale \n".format(a=round(tff/P,4)))
file.write("There are {a} collisions happened during the simulation \n".format(a=sim.collisions_Nlog))
file.write("---------- \n")
file.write("Collisional cross-section of the real particle: {a} m2 \n".format(a=s_cr1))
file.write("Radius of a particle: {a} km \n".format(a=a1/1e3))
file.write("Mass of a particle: {a} kg \n".format(a=mp1))
file.close()