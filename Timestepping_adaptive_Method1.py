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
k = 0.05 #Dissipation term
theta = 1  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicholson
msh = create_unit_square(MPI.COMM_WORLD,48,48, CellType.triangle)
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

# Step in time
t = 0.0
dt = 5.0e-5  # time step
#  Reduce run time if on test (CI) server
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 1 * dt
else:
    T =80000*dt
dk = dolfinx.fem.Constant(msh, dt)
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
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

from datetime import datetime
startTime = datetime.now()
print("------------------------------------")
print("Simulation Start")
print("------------------------------------")

step = "Evolve"

# Output file
file = XDMFFile(MPI.COMM_WORLD, "adpttest3_1.5s.xdmf", "w")
file.write_mesh(msh)

# Get the sub-space for c and the corresponding dofs in the mixed space
# vector
c = u.sub(0)
u0.x.array[:] = u.x.array
ii = 0.0
bisection_count = 0
max_incr = 3
incr = 0
while (t < T):
    t += float(dk)
    ii +=1
    (iter, converged) = solver.solve(u)
    file.write_function(c, t)
    if converged:
        u.x.scatter_forward()
        u0.x.array[:] = u.x.array
        if ii % 1 ==0:
            now = datetime.now()
            print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
            print(float(t))
        if ii>6500 and incr < max_incr:    
            if iter <= 3:
                dt = 1.5 * dt
                print (float(dt))
                dk.value = dt
                incr += 1
            
        elif iter > 5:
            dt = dt / 2
            dk.value = dt

        # Reset Biseciton Counter
        bisection_count = 0

    else:
        # Break the loop if solver fails too many times
        bisection_count += 1

        if bisection_count > 5:
            print("Error: Too many bisections")
            break

        print("Error Halfing Time Step")
        t = t - float(dk)
        dt = dt / 2
        dk.value = dt
        print(f"New Time Step: {dt}")
        u.x.array[:] = u0.x.array

file.close()






