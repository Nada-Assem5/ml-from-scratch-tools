"""
Required Libraries:
Sympy
"""

import sympy as sp

def Lagrange_Equality(obj_fun, constraints, n_vars = 1, minimize= True):
  """
  Implements Lagrange Multipliers approach for solving equality constrained
  univariate and multivariate optimization problems.

  Parameters:
      obj_fun (string): Objective function in variables x_0, x_1, and so on.
                          Example: '2 * x_0 + x_1 + 10'
      constraints (list of strings): List of equality constraints set to 0.
                          Example: ['x_0 + 2 * x_1**2 - 3']
      n_vars (int): Number of decision variables. Default is 1.
      minimize (bool): True for minimization, False for maximization. Default is True.

  Returns:
        tuple(list of floats, float): ([best_point], best_value) rounded to 4 decimal points.
                          Example: ([2.9688, 0.125], 16.0625)
  """
  x_vars = sp.symbols(f'x_0:{n_vars}')
  obj_fun = sp.sympify(obj_fun)
  cons = [sp.sympify(c) for c in constraints ]
  lambdas = sp.symbols(f"lambda_0:{len(cons)}")

  L = obj_fun

  for i in range(len(cons)):
    L = L - lambdas[i] * cons[i]

  equations = []

  for v in x_vars:
      equations.append(sp.diff(L, v))

  for l in lambdas:
      equations.append(sp.diff(L, l))


  solutions = sp.solve(equations, list(x_vars) + list(lambdas))

  best_point = None
  best_value = None

  if isinstance(solutions, dict):
         solutions = [solutions]

  for sol in solutions:
      if isinstance(sol, dict):
              point = [sol[v] for v in x_vars]
      else:
            point = list(sol[:n_vars])

      value = float(obj_fun.subs([(x_vars[i], point[i]) for i in range(n_vars)]))

      if best_value is None:
                best_value = value
                best_point = point

      elif minimize and value < best_value:
                best_value = value
                best_point = point

      elif not minimize and value > best_value:
                best_value = value
                best_point = point

  best_point = [round(float(p), 4) for p in best_point]
  best_value = round(best_value, 4)

  return (best_point, best_value)