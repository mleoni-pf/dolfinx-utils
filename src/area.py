from ufl import (Constant, ds, tetrahedron)

a = Constant(tetrahedron)

area = a * ds(0)

forms = [area]
