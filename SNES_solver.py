import os
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element, mixed_element
from dolfinx.io import XDMFFile
import numpy as np
import ufl
from dolfinx import la
from dolfinx.fem import (Function, form, functionspace)
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector,
                               create_matrix)
from dolfinx.mesh import create_unit_square, CellType
from ufl import TrialFunction, derivative, dx, grad, inner


class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bc):
        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bc = bc
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        assemble_matrix(J, self.a)
        # assemble_matrix(J, self.a, bcs=[self.bc])
        J.assemble()


lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06  # time step
theta = 0.5  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicholson
msh = create_unit_square(MPI.COMM_WORLD, 24, 24, CellType.triangle)
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

# Compute the chemical potential df/dc
c = ufl.variable(c)
f = 100 * c**2 * (1 - c)**2
dfdc = ufl.diff(f, c)

# mu_(n+theta)
mu_mid = (1.0 - theta) * mu0 + theta * mu

# Weak statement of the equations
F0 = inner(c, q) * dx - inner(c0, q) * dx + \
    dt * inner(grad(mu_mid), grad(q)) * dx
F1 = inner(mu, v) * dx - inner(dfdc, v) * dx - \
    lmbda * inner(grad(c), grad(v)) * dx

F = F0 + F1
bc = []
problem = NonlinearPDE_SNESProblem(F, u, bc)

# Interpolate initial condition
np.random.seed(30)
u.sub(0).interpolate(lambda x: 0.25 + 0.01 *
                     (0.5 - np.random.rand(x.shape[1])))
u.x.scatter_forward()

b = la.create_petsc_vector(ME.dofmap.index_map, ME.dofmap.index_map_bs)
J = create_matrix(problem.a)

# Create Newton solver and solve
snes = PETSc.SNES().create()
opts = PETSc.Options()
opts['snes_monitor'] = None
snes.setFromOptions()
snes.setType('vinewtonrsls')  # Changed even to vinewtonssls
snes.setFunction(problem.F, b)
snes.setJacobian(problem.J, J)
snes.setTolerances(rtol=1.0e-8, max_it=50)

snes.getKSP().setType("preonly")
snes.getKSP().setTolerances(rtol=1.0e-8)
snes.getKSP().getPC().setType("lu")

# Setting constraint for concentration
Fun1, dof1 = ME.sub(0).collapse()
c_min = Function(ME)
c_max = Function(ME)
c_min.sub(0).interpolate(lambda x: np.zeros(x.shape[1], dtype=np.float64))
c_max.sub(0).interpolate(lambda x: np.ones(x.shape[1], dtype=np.float64))
c_min.sub(1).interpolate(lambda x: np.full(x.shape[1], -1e7, dtype=np.float64))
c_max.sub(1).interpolate(lambda x: np.full(x.shape[1], 1e7, dtype=np.float64))
snes.setVariableBounds(c_min.vector, c_max.vector)
# Output file
file = XDMFFile(MPI.COMM_WORLD, "SNES.xdmf", "w")
file.write_mesh(msh)

# Step in time
t = 0.0
#  Reduce run time if on test (CI) server
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 3 * dt
else:
    T = 500 * dt
# Get the sub-space for c and the corresponding dofs in the mixed space
# vector
u0.x.array[:] = u.x.array

while (t < T):
    t += dt
    r = snes.solve(None, u.vector)
    # print(f"Step {int(t/dt)}: num iterations: {r[0]}")
    u0.x.array[:] = u.x.array
    file.write_function(u.sub(0), t)
file.close()