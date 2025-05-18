import numpy as np

"""
    ███████╗██╗   ██╗██████╗  ██████╗ ██╗  ██╗██╗   ██╗
    ██╔════╝██║   ██║██╔══██╗██╔═══██╗██║ ██╔╝██║   ██║
    ███████╗██║   ██║██║  ██║██║   ██║█████╔╝ ██║   ██║
    ╚════██║██║   ██║██║  ██║██║   ██║██╔═██╗ ██║   ██║
    ███████║╚██████╔╝██████╔╝╚██████╔╝██║  ██╗╚██████╔╝
    ╚══════╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ 
                                                                                             
     ██████╗ ███████╗███╗   ██╗                        
    ██╔════╝ ██╔════╝████╗  ██║                        
    ██║  ███╗█████╗  ██╔██╗ ██║                        
    ██║   ██║██╔══╝  ██║╚██╗██║                        
    ╚██████╔╝███████╗██║ ╚████║                        
     ╚═════╝ ╚══════╝╚═╝  ╚═══╝ 
    
    An "efficient" Sudoku Generator derived from Chapter 6 of 
    "Artificial Intelligence: A Modern Approach"
    
    A one-shot upgrade to my 2020 java solver.
        
    Version 1.0
"""

# Constants.
VEC_SIZE = 9
DOM_SIZE = 81
SUB_SIZE = 3

# Convert a point to a domain index.
def point_to_domain(i, j):
    return i * VEC_SIZE + j

# Convert a domain index to a point.
def domain_to_point(x):
    return x // VEC_SIZE, x % VEC_SIZE

"""
Class: Grid

A Sudoku Grid represented with a mailbox and a list of domains.
"""
class Grid:

    # Constructor.
    def __init__(self):
        # Unary constraint applied to all domains. Value must be an integer in [1,9].
        self.grid    = [[0 for _ in range(VEC_SIZE)] for __ in range(VEC_SIZE)]
        self.domains = [[_ + 1 for _ in range(VEC_SIZE)] for __ in range(DOM_SIZE)]
        self.step = 0

        # Initialize first row to generate one of 9! grids.
        r = np.arange(1, 10)
        np.random.shuffle(r)
        self.grid[0] = r.tolist()
        for i in range(VEC_SIZE):
            self.domains[i] = [self.grid[0][i]]

    # Get arcs to neighbors.
    def get_arcs(self, i, j, arcs, exclude):
        cell = point_to_domain(i, j)
        start_r = i * VEC_SIZE
        start_c = j
        start_s = point_to_domain(i - (i % SUB_SIZE), j - (j % SUB_SIZE))

        # 1-9
        for o in range(start_r, start_r + VEC_SIZE):
            if o != cell and o != exclude:
                arcs.append((o, cell))
            if start_c != cell and start_c != exclude:
                arcs.append((start_c, cell))
            if start_s != cell and start_s != exclude:
                arcs.append((start_s, cell))

            start_s += 1 + (6 & -(o % SUB_SIZE == 2))
            start_c += VEC_SIZE

    # is the assignment at (i, j) consistent?
    def is_consistent_assignment(self, i, j):
        ri = i - (i % 3)
        rj = j - (j % 3)

        row = col = sub = cnt = 0
        for k in range(9):
            # Binary constraint.
            # Rows, cols, sub-grids.
            row += self.grid[i][k] == self.grid[i][j]
            col += self.grid[k][j] == self.grid[i][j]

            k3 = (k % 3)
            sub += self.grid[ri + cnt][rj + k3] == self.grid[i][j]
            cnt += k3 == 2

        return row < 2 and col < 2 and sub < 2

    # Revision step of AC-3.
    def revise(self, x, y):
        rev = False
        old = self.domains[x]
        new = []
        for o in old:
            sats = False

            for p in self.domains[y]:
                # Binary constraint.
                # Rows, cols, sub-grids.
                if o != p:
                    sats = True

            if sats:
                new.append(o)
            else:
                rev = True

        self.domains[x] = new

        if len(new) == 1:
            i, j = domain_to_point(x)
            self.grid[i][j] = new[0]

        return rev, (x, old)

    # do inference by AC-3.
    def do_inference(self, i, j, arcs):
        q = arcs.copy()
        restore = []
        while len(q) > 0:
            x, y = q[0]
            q = q[1:]

            rev, old = self.revise(x, y)

            if not rev:
                continue

            restore.append(old)

            if len(self.domains[x]) <= 0:
                return False, restore

            i, j = domain_to_point(x)
            self.get_arcs(i, j, q, y)

        return True, restore

    # Undo assignments made during inference.
    def undo_inference(self, restore):
        for x, old in restore:
            if len(old) > 1:
                i, j = domain_to_point(x)
                self.grid[i][j] = 0
            self.domains[x] = old

    # Score the values.
    def score_values(self, dom, arcs):
        cnt = [0 for _ in range(len(dom))]
        for i in range(len(dom)):
            v = dom[i]
            for x, _ in arcs:
                sat = False
                for vx in self.domains[x]:
                    if v != vx:
                        sat = True

                if not sat:
                    cnt[i] += 1

        return cnt

    # Get the next value by incremental
    # selection sort.
    def next_value(self, dom, score, idx):
        idx += 1
        if idx >= len(dom):
            return 0, idx

        mx = score[idx]
        x = idx
        for i in range(idx, len(dom)):
            if score[i] > mx:
                x = i
                mx = score[x]

        value = dom[x]
        dom[x] = dom[idx]
        dom[idx] = value
        score[x] = score[idx]

        return value, idx

    # Get the next domain.
    def next_domain(self):
        mn = (1 << 30)
        next_dom = DOM_SIZE
        for i in range(DOM_SIZE):
            if len(self.domains[i]) <= 1:
                continue

            if len(self.domains[i]) < mn:
                mn = len(self.domains[i])
                next_dom = i

        return next_dom

    # Check a complete assignment for
    # consistency.
    def check_consistency(self):
        for i in range(VEC_SIZE):
            for j in range(VEC_SIZE):
                if not self.is_consistent_assignment(i, j):
                    print(i, j)
                    return False
        print("consistent")
        return True

    # Solve the Sudoku CSP with backtracking.
    def backtrack(self):
        # Use MRV (Minimum Remaining Values)
        # to order domains.
        x = self.next_domain()
        if x == DOM_SIZE:
            return True

        # Print step count.
        self.step += 1
        print(f"step {self.step}")

        # Get the domain.
        dom = self.domains[x]
        i, j = domain_to_point(x)

        # Get arcs to neighbors.
        arcs = []
        self.get_arcs(i, j, arcs, DOM_SIZE)

        # Score the values of the domain using
        # its neighbors, for incremental sorting.
        score = self.score_values(dom, arcs)

        # Iterate through the values.
        idx = -1
        while True:
            # Use LCV (Least Constraining Value) to order
            # values.
            o, idx = self.next_value(dom, score, idx)
            if o == 0:
                break

            # Assign
            self.grid[i][j] = o
            self.domains[x] = [o]

            # Check that the assignment is consistent.
            if not self.is_consistent_assignment(i, j):
                continue

            # Do inference, in our case MAC
            # (Maintaining Arc Consistency via AC-3).
            infer, restore = self.do_inference(i, j, arcs)
            if not infer:
                # Undo assignments if inference failed.
                self.undo_inference(restore)
                continue

            # Recursively assign.
            if self.backtrack():
                return True

            # Undo assignments made during inference
            # if backtracking.
            self.undo_inference(restore)

        # Undo assignments if backtracking.
        self.domains[x] = dom
        self.grid[i][j] = 0
        return False

'''
MAIN
'''
if __name__ == "__main__":
    g = Grid()
    g.backtrack()
    g.check_consistency()

    k = 0
    for r in g.grid:
        i = 0
        print(end=' ')
        for o in r:
            print(o, end=' ')
            if i != 8 and i % SUB_SIZE == 2:
                print('|', end=' ')
            i += 1
        print()
        if k != 8 and k % SUB_SIZE == 2:
            print("-------+-------+-------")
        k += 1