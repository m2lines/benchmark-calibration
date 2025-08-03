import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
import scipy

class Simplex:
    def __init__(self, vertices=None, depth=0, parent=None, dimension=2):
        # Shape of vertices
        if vertices is None:
          # (dimension+1, dimension)
          # Here, each row is a d-dimensional vector
          # We use standard orthonormal basis and origin of coordinates
          vertices = np.vstack([np.zeros(dimension),np.eye(dimension)])
          
        self.vertices = vertices
        self.children = []
        self.parent = parent
        self.depth = depth
        self.dimension = dimension

    def longest_edge(self):
        max_length = -np.inf
        longest = None
        # All unordered pairs of vertex indices
        # We pick the first longest edge for simplicity
        # even though we have a few of them
        # See https://doi.org/10.1111/j.1467-8659.2011.01853.x
        # for possible options. Here we hope to follow the
        # simplest possible algorithm
        for i, j in combinations(range(len(self.vertices)), 2):
            v1, v2 = self.vertices[i], self.vertices[j]
            length = np.linalg.norm(v2 - v1)
            if length > max_length:
                max_length = length
                longest = (v1, v2)
        return longest

    def bisect(self):
        '''
        Here we form only two new simplexes
        '''
        v1, v2 = self.longest_edge()
        M = 0.5 * (v1 + v2)  # midpoint of longest edge

        # In the first simplex we replace 
        # v2 with M
        idx_v2 = [idx for idx in range(len(self.vertices)) if np.allclose(self.vertices[idx],v2)][0]
        vertices1 = self.vertices.copy()
        vertices1[idx_v2] = M

        # In the second simplex we replace
        # v1 with M
        idx_v1 = [idx for idx in range(len(self.vertices)) if np.allclose(self.vertices[idx],v1)][0]
        vertices2 = self.vertices.copy()
        vertices2[idx_v1] = M

        simplex1 = Simplex(vertices1, depth=self.depth+1, parent=self, dimension=self.dimension)
        simplex2 = Simplex(vertices2, depth=self.depth+1, parent=self, dimension=self.dimension)
        self.children = [simplex1, simplex2]
        return simplex1, simplex2

    def get_key(self):
        '''
        This function assings a unique value to each tringle
        up to permutation of vertices.
        '''
        # Each vertex is a row in the self.vertices array
        # Convert each vertex (1D array) into a tuple
        vertex_tuples = [tuple(v) for v in self.vertices]
        
        # Sort to ensure consistent ordering regardless of permutation
        sorted_vertices = tuple(sorted(vertex_tuples))
        
        return sorted_vertices

    def is_leaf(self):
        return len(self.children) == 0

    def iterate_depth(self, depth):
      if self.depth == depth:
        yield self
      
      for child in self.children:
        yield from child.iterate_depth(depth)

    def plot(self, min_depth=0):
      if self.dimension == 2:
        
        triangle = np.vstack([self.vertices, self.vertices[0]])
        line = plt.plot(triangle[:, 0], triangle[:, 1], linewidth=2)  # triangle edges
        try:
          color = line[0].get_color()
          plt.plot(self.x_solved[0], self.x_solved[1], color=color, ls='', marker='o', markerfacecolor='none', markersize=15)
        except:
          pass

      elif self.dimension == 3:
        ax = plt.gca()
        # Part 1: Gather all line segment coordinates
        Vx, Vy, Vz = [], [], []
        for i, j in combinations(range(len(self.vertices)), 2):
            p1, p2 = self.vertices[i], self.vertices[j]
            Vx += [p1[0], p2[0], np.nan]
            Vy += [p1[1], p2[1], np.nan]
            Vz += [p1[2], p2[2], np.nan]

        # Part 2: Plot all lines in one command using the same color
        line = ax.plot(Vx, Vy, Vz, linewidth=2)
        
        try:
          color = line[0].get_color()
          ax.plot(self.x_solved[0], self.x_solved[1], self.x_solved[2], marker='o', ls='', markersize=15, color=color)
        except:
           pass
        # Plot vertices
        ax.plot(*self.vertices.T, color='r', ls='', markersize=10, marker='o')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if min_depth == 0:
          ax.set_xlim([0,1])
          ax.set_ylim([0,1])
          ax.set_zlim([0,1])
        ax.set_box_aspect([1, 1, 1])
      else:
        raise NotImplementedError

    def plot_all(self, min_depth=0):
      if self.depth>=min_depth:
        self.plot(min_depth)
      
      if self.is_leaf():
          plt.gca().set_aspect('equal')
          return
      else:
          for child in self.children:
              child.plot_all(min_depth)

    def x_to_baricentric(self, x):
      '''
      Here we convert point of R^dimension parameter
      space into baricentric coordinates
      to answer the question whether it is inside the simplex
      Point x in baricentric coordinates has the form:
      x = \sum coeff_i * self.vertices[i]
      \sum coeff_i = 1
      where coeff_i are scalars
      '''
      M = np.hstack([self.vertices, np.ones([len(self.vertices),1])])
      rhs = np.array([*x, 1])
      #coeff = np.linalg.solve(M.T, rhs)
      coeff, *_ = np.linalg.lstsq(M.T, rhs, rcond=None)
      #print(coeff@self.vertices - x) 
      return coeff

    def is_x_inside(self, x, tol=-1e-3):
      '''
      This function simply answers a question
      wheather point x of R^dimension parameter 
      space is inside the triangle or not

      Here parameter tol defines how to treat points
      which are close to edges. If tol is positive,
      edge points are excluded within this distance.
      Instead, if tol is negative, edge points are included
      within this distance.
      '''
      coeff = self.x_to_baricentric(x)
      return np.all(coeff >= tol) and np.all(coeff <= 1 - tol)

    def return_flipped_simplex(self, x, tol=-1e-3):
      '''
      Given point x, flip the simplex by reflecting the vertex
      with most negative barycentric coordinate across the opposite face.
      '''
      coeff = self.x_to_baricentric(x)  # (d+1,) array
      if np.min(coeff) > tol:
          return None  # No flip needed

      idx = np.argmin(coeff)  # Index of vertex to flip
      v_bad = self.vertices[idx]
      face_vertices = np.delete(self.vertices, idx, axis=0)  # Opposite face (d points in R^d)

      # Compute face centroid
      face_center = np.mean(face_vertices, axis=0)

      # Reflect v_bad across the face centroid
      v_flipped = 2 * face_center - v_bad

      # Construct new simplex with flipped vertex
      new_vertices = face_vertices.tolist()
      new_vertices.insert(idx, v_flipped)  # Insert at same index for consistent ordering
      new_vertices = np.array(new_vertices)

      # Return new simplex object (e.g., your subclass of Simplex)
      flipped_simplex = Simplex(new_vertices, depth=self.depth, parent=self, dimension=self.dimension)
      self.children.append(flipped_simplex)
      return flipped_simplex

    def find_containing_ancestor_and_then_child(self, x_solve):
      """
      Step 1: Traverse upward to find the lowest ancestor that contains x_solve.
      Step 2: Recursively bisect and descend until the depth of 'self' is reached,
              always choosing the child that contains x_solve.
      """
      # Step 1: Find lowest ancestor that contains x_solve
      current = self
      lowest = None
      while current is not None:
          # Here we want to make sure that ancestor
          # indeed contains the points
          if current.is_x_inside(x_solve):
              lowest = current
          current = current.parent

      if lowest is None:
          #print(f"No ancestor contains {x_solve}")
          return None

      # Step 2: Descend using bisection until reaching target depth
      while lowest.depth < self.depth:
          if not lowest.children:
              lowest.bisect()  # This should populate lowest.children

          for child in lowest.children:
              if child.is_x_inside(x_solve):
                  lowest = child
                  break
          else:
            #print(f"Bisection failed to produce a child containing {x_solve} at depth {lowest.depth+1} \n")
            #   print('Lowest simplex', lowest.vertices, lowest.is_x_inside(x_solve))
            #   print('Lowest simplex: barycentric coords', lowest.x_to_baricentric(x_solve))
            #   for child in lowest.children:
            #     print('Children simplex', child.vertices, child.is_x_inside(x_solve))
            #     print('Children simplex: barycentric coords', child.x_to_baricentric(x_solve))
            #   print('Target simplex', self.vertices, self.is_x_inside(x_solve))
            #   print('Target simplex: barycentric coords', self.x_to_baricentric(x_solve))

            #   print("\n")

              return None

      #print(f"Found refined simplex at depth {lowest.depth} with vertices {lowest.vertices}")
      return lowest

    def interpolation_weights(self, F):
      '''
      This function recieves matrix F
      of shape [dimension+1, arbitrary_big_number]
      This is a vector-valued function evaluated
      in vertices of the simplex
      The dimensionality along the second index
      (what makes it vector function)
      represents the dimensionality of the observational
      space and scales really well.

      We are looking for a linear function w.r.t. x
      which intersects vertices of the simplex exactly.
      We note that for each component of the vector-valued
      function we construct a separate linear 
      interpolant.

      The free parameters of the linear function
      have the following dimensionality:
      A [dimension+1, arbitraty_big_number]
      That is dimension+1 free parameters per each
      vector component out of arbitrary_big_number

      This matrix acts on the coordinates of the 
      vertices enlarged with a constant vector
      X [dimension+1, dimension+1]

      Thus, we have the following system to solve:
      F = X@A

      It's solution can be found as
      A = X^{-1}@F
      '''

      X = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
      return np.linalg.solve(X,F)

    def solve_inverse_problem(self, F):
      '''
      This function recieves matrix F
      of shape [dimension+1, arbitrary_big_number]
      This is a vector-valued function evaluated
      in vertices of the simplex

      The goal of this function is to find the minimum of
      ||f(x,y)||^2
      where f(x,y) is replaced with linear interpolation
      of the local error function which has the form
      error(x,y)  = [x;1].T@A
      Here x is a vector of the cartesian coordinates
      of size dimension to be determined.
      
      To solve equation in least squares we need to split matrix 
      into two parts:
      [x;1].T @ [A[:-1]; A[-1]] = x@A[:-1] + A[-1] = error
      We solve the problem in least squaresL
      x.T @ A[:-1] @ A[:-1].T = - A[-1] @ A[:-1].T

      Transposing, we have
      A[:-1] @ A[:-1].T @ x = - A[:-1] @ A[-1].T
      this is simply
      x = - np.linalg.pinv(A[:-1].T) @ A[-1].T)
      '''
      A = self.interpolation_weights(F)
      A_upper = A[:-1]
      A_lower = A[-1]

      if (np.linalg.cond(A_upper@A_upper.T) > 1e+6):
        print('Warning: Singular matrix. Multiple solutions exist. Closest to the origin is returned')

      x_solved = - np.linalg.pinv(A_upper.T) @ A_lower.T

      self.x_solved = x_solved
      self.F = F
      return x_solved
    
def SimplexBisection(forward_map, max_depth=1, exploration=False, root=None, dimension=None, bisect_parent=False, evaluate_best=False):
  '''
  This function assumes that the solution to inverse
  problem ||f(x,y)||_2 -> min is inside the
  initial simplex. (exploration option loosens this assumption)
  We iteratively make sure that under linear assumption
  solution is indeed inside a given simplex
  and bisect it.
  Then we check two bisected simplexes and bisect
  only those where solution is expected to be
  '''

  if root is None:
    root = Simplex(dimension=dimension)

  seen_simplexes = set(root.get_key())
  def bisect_simplexes_guided_by_loss(simplexes, force_bisect=False, verbose=False):
    # This key is activated when all triangles do 
    # not have a solution inside
    nothing_inside = True
    for simplex in simplexes:
      # Evaluate vector-valued function in simplex vertices
      F = np.vstack([forward_map.lookup_table(tuple(simplex.vertices[i])) for i in range(len(simplex.vertices))])
      # Solve the inverse problem
      x_solve = simplex.solve_inverse_problem(F)
      if evaluate_best:
        forward_map.lookup_table(tuple(x_solve))
      # Check that the solution lies inside
      is_inside = simplex.is_x_inside(x_solve)
      nothing_inside = nothing_inside and not is_inside
      if max_depth >= 0:
        if is_inside or force_bisect:
          if simplex.depth >= max_depth:
            continue
          S1, S2 = simplex.bisect()
          S1.origin = 'created by main algorithm'
          S2.origin = 'created by main algorithm'
          seen_simplexes.add(S1.get_key())
          seen_simplexes.add(S2.get_key())
          bisect_simplexes_guided_by_loss([S1, S2])

    if nothing_inside and bisect_parent:
      if verbose:
        print("bisect_parent is triggered at depth:", [s.depth for s in simplexes])
      new_childs = []
      for simplex in simplexes:
        # Evaluate vector-valued function in simplex vertices
        F = np.vstack([forward_map.lookup_table(tuple(simplex.vertices[i])) for i in range(len(simplex.vertices))])
        x_solve = simplex.solve_inverse_problem(F)
        new_child = simplex.find_containing_ancestor_and_then_child(x_solve)
        if new_child is not None:
          if new_child.get_key() not in seen_simplexes:
            seen_simplexes.add(new_child.get_key())
          new_child.origin = 'created by bisect_parent'
          new_childs.append(new_child)
    
      if len(new_childs) > 0:
        nothing_inside = False
        bisect_simplexes_guided_by_loss(new_childs, force_bisect=False)
      else:
        nothing_inside = True
    
    # The last stage which moves us away from original tree structure
    if exploration and nothing_inside:
      if verbose:
        print("Exploration triggered at depth:", [s.depth for s in simplexes])
      for simplex in simplexes:
        F = np.vstack([forward_map.lookup_table(tuple(simplex.vertices[i])) for i in range(len(simplex.vertices))])
        x_solve = simplex.solve_inverse_problem(F)
        S = simplex.return_flipped_simplex(x_solve)
        if S is not None:
          if S.get_key() not in seen_simplexes:
            seen_simplexes.add(S.get_key())
            bisect_simplexes_guided_by_loss([S])
          else:
            continue

  bisect_simplexes_guided_by_loss([root])
  return root, seen_simplexes