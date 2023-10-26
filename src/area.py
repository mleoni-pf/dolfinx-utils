from basix.ufl import element
from ufl import (Constant, ds, Mesh)

dim = 3

coord_element = element("Lagrange", "tetrahedron", 1, shape=(dim,))
mesh = Mesh(coord_element)

a = Constant(mesh)

area = a * ds(0)

forms = [area]
