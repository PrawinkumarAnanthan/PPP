import os
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot
from dolfinx.fem import Function, functionspace,assemble_scalar
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner
import dolfinx

# Save all logging to file
#log.set_output_file("log.txt")
epsilon = 1/64  # interface thickness
areainteng = 0.3 #Area specific interface energy
dt = 5.0e-06  # time step
k = 0.05 #Dissipation term
theta = 0.5  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicholson
msh = create_unit_square(MPI.COMM_WORLD,24,24, CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1)
ME = functionspace(msh, mixed_element([P1, P1]))
q, v = ufl.TestFunctions(ME)
u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step
# Split mixed functions
c, mu = ufl.split(u)
c0, mu0 = ufl.split(u0)
# Zero u
u.x.array[:] = 0.0
# Interpolate initial condition
np.random.seed(30)
u.sub(0).interpolate(lambda x: 0.25  + 0.02 * (0.5 - np.random.rand(x.shape[1])))
#print (u.x.array)
u.x.scatter_forward()

# Compute the chemical potential df/dc
c = ufl.variable(c)
f = areainteng*((6*(1/epsilon)*c**2*(1-c)**2))
dfdc = ufl.diff(f, c)
# mu_(n+theta)  
mu_mid = (1.0 - theta) * mu0 + theta * mu

# Weak statement of the equations
F0 = inner(c, q) * dx - inner(c0, q) * dx + dt * k * inner(grad(mu_mid), grad(q)) * dx
F1 = inner(mu, v) * dx - inner(dfdc, v) * dx - 3*epsilon*areainteng*inner(grad(c), grad(v)) * dx

F = F0 + F1   

# Create nonlinear problem and Newton solver
problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2

# We can customize the linear solver used inside the NewtonSolver by
# modifying the PETSc options
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

# Output file
file = XDMFFile(MPI.COMM_WORLD, "demotest_ch/0.25hc2425k.xdmf", "w")
file.write_mesh(msh)

# Step in time
t = 0.0

#  Reduce run time if on test (CI) server
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 1 * dt
else:
    T =90000 * dt

# Get the sub-space for c and the corresponding dofs in the mixed space
# vector
V0, dofs = ME.sub(0).collapse()
# val = dolfinx.fem.Function(V0)
c = u.sub(0)
u0.x.array[:] = u.x.array
E_stored =[0]
timesteps =[0]
previous_energy = None  # Initialize previous_energy variable

while (t < T):
    t += dt
    r = solver.solve(u)
    print(f"Step {int(t/dt)}: num iterations: {r[0]}")
    
    u0.x.array[:] = u.x.array

    #print(val.x.array)
    # energy = dolfinx.fem.form(f*dx)
    # current_energy = assemble_scalar(energy)
    # E_stored.append(current_energy)
    # timesteps.append(t)
    # #c, mu = ufl.split(u)
    # val.x.array[:] = u.x.array[dofs]
    # val.x.scatter_forward()
    #print(u.x.array)

    # # Check if the current energy is the same as the previous one
    # if previous_energy is  not None and current_energy == previous_energy:
    #     print("Breaking loop because energies are the same.")
    #     break

    # # # Update previous_energy for the next iteration
    # previous_energy = current_energy
    file.write_function(c, t)

file.close()

#print(u0.x.array)
#print (E_stored)
#print(val.x.array)

# #For saving values of concentration in file
# file_path = './topopteqn_pvalue.txt'
# np.savetxt(file_path,val.x.array)

# #Plotting 
# import matplotlib.pyplot as plt
# plt.plot(timesteps, E_stored, label='Energy vs. Time')
# plt.xlabel('Time')
# plt.ylabel('Energy')
# plt.title('Energy vs. Time')
# plt.legend()
# plt.show()
