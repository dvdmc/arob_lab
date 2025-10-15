import numpy as np
from scipy.interpolate import make_interp_spline, PPoly

class TrajectoryGenerator:
    """
    A class to generate smooth spline trajectories in 3D space (x, y, z)
    given a list of waypoints and corresponding times.
    """

    def __init__(self):
        """
        Initialize internal variables for storing waypoints and spline parameters.
        """
        self.waypoints = []        # list of [x, y, z]
        self.times = []            # list of time stamps
        # General constraints at knots (waypoints):
        # { d (derivative order) : { k (knot index 0..M-1) :
        #     {"value": np.array([dx,dy,dz]), "side": "left"|"right"|"both"} } }
        self.waypoint_constraints = {}
        self.trajectory = None     # list[PPoly_x, PPoly_y, PPoly_z]

    def add_waypoint(self, waypoint, time):
        """
        Add a new waypoint to the trajectory.

        Args:
            waypoint (list or np.ndarray): 3D waypoint [x, y, z]
            time (float): Time corresponding to this waypoint
        """
        waypoint = np.asarray(waypoint, dtype=float)

        if waypoint.shape != (3,):
            raise ValueError("Waypoint must be a 3D vector: [x, y, z].")

        if self.times and time <= self.times[-1]:
            raise ValueError("Time for new waypoint must be greater than the last one.")

        self.waypoints.append(waypoint)
        self.times.append(float(time))

    
    def add_waypoint_constraint(self, derivative_order: int, wpt_index: int,
                        value, side: str = "auto"):
        """
        Constrain the d-th time derivative at a waypoint time t_k (k in [0..M-1]).

        Args:
            derivative_order (int): 0=position, 1=velocity, 2=acceleration, 3=jerk, ...
            wpt_index (int): waypoint index (0..M-1).
            value (array-like len 3): [dx, dy, dz]. Use np.nan where you don't want to constrain an axis.
            side (str): 'left' (use segment k-1, tau=T_{k-1}), 'right' (use seg k, tau=0),
                        'both', or 'auto' (left for interior k>0, right for k=0).
                        Note: For interior knots and d <= continuity_order, one side is enough
                        (continuity will mirror). For d > continuity_order, consider 'both'.
        """
        if derivative_order < 0:
            raise ValueError("derivative_order must be >= 0.")
        if not self.times:
            raise RuntimeError("Add waypoints before setting knot constraints.")
        M = len(self.times)
        if not (0 <= wpt_index <= M - 1):
            raise ValueError("wpt_index out of range.")
        v = np.asarray(value, dtype=float)
        if v.shape != (3,):
            raise ValueError("value must be a 3D vector [dx, dy, dz] (use np.nan to skip an axis).")

        if side not in ("left", "right", "both", "auto"):
            raise ValueError("side must be 'left', 'right', 'both', or 'auto'.")
        if side == "auto":
            side = "right" if wpt_index == 0 else "left"

        self.waypoint_constraints.setdefault(derivative_order, {})[wpt_index] = {
            "value": v, "side": side
        }

    
    def generate_trajectory(self, opts):
        """
        Build the trajectory according to a single 'opts' struct/dict.

        Expected fields in 'opts' (case-insensitive for trajectory_type):
            - trajectory_type: one of
                "piecewise_polynomial"        -> calls compute_piecewise_polynomial(order)
                "minimum_snap"                -> calls compute_minimum_snap(order, continuity_order, reg)
                "minimum_snap_normalized"     -> calls compute_minimum_snap_normalized(order, continuity_order, reg)
            - order (int):          degree per segment (where applicable). Defaults vary by method.
            - continuity_order (int): continuity across knots (only for minimum_*). Default 3.
            - reg (float):          small Tikhonov regularization (only for minimum_*). Default 1e-9.

        Notes:
            - 'opts' can be a dict or any object with attributes (e.g., SimpleNamespace).
            - Derivative constraints must be added beforehand via:
            add_waypoint_constraint(derivative_order, waypoint_index, value, side)
        """
        # Helper to get keys from dict-like or attribute-like objects
        def _get(opt_obj, key, default=None):
            if opt_obj is None:
                return default
            if isinstance(opt_obj, dict):
                return opt_obj.get(key, default)
            # Fallback to attribute access
            return getattr(opt_obj, key, default)

        ttype = _get(opts, "trajectory_type", None)
        if ttype is None:
            raise ValueError("opts.trajectory_type is required. "
                             "Valid values: 'piecewise_polynomial', 'minimum_snap'.")

        ttype = str(ttype).strip().lower()

        if ttype == "piecewise_polynomial":
            order = int(_get(opts, "order", 3))
            self.compute_piecewise_polynomial(order=order)
            return
        
        if ttype == "minimum_snap":
            order = int(_get(opts, "order", 7))
            continuity_order = int(_get(opts, "continuity_order", 3))
            reg = float(_get(opts, "reg", 1e-9))
            # Always use the normalized implementation
            self.compute_minimum_snap(order=order,
                                    continuity_order=continuity_order,
                                    reg=reg)
            return

        raise ValueError(f"Unknown trajectory_type='{ttype}'. "
                         "Valid values: 'piecewise_polynomial', 'minimum_snap'.")

    def evaluate_trajectory(self, t: float, derivative_order: int = 0) -> np.ndarray:
        """
        Evaluate the 3D trajectory or any of its time derivatives at a given time.

        Args:
            t (float): Time to evaluate.
            derivative_order (int): Order of the time derivative to evaluate.
                0 -> position
                1 -> velocity
                2 -> acceleration
                3 -> jerk
                etc.

        Returns:
            np.ndarray: 3D vector [x, y, z] of the evaluated derivative.
        """
        if self.trajectory is None:
            raise RuntimeError(
                "Trajectory has not been computed. Call generate_trajectory(...) first."
            )

        # Clamp time to the valid range
        t0, tf = self.times[0], self.times[-1]
        tt = float(np.clip(t, t0, tf))

        # Evaluate derivative of each component's spline
        values = []
        for s in self.trajectory:
            deriv = s.derivative(derivative_order) if derivative_order > 0 else s
            values.append(float(deriv(tt)))
        return np.array(values, dtype=float)

    def sample_trajectory(self, deltat):
        """
        Uniformly sample the spline trajectory at fixed intervals.

        Args:
            deltat (float): Sampling period (seconds). Must be > 0.

        Returns:
            dict: 1D arrays with positions, velocities, and accelerations:
                {
                    'x','y','z',
                    'vx','vy','vz',
                    'ax','ay','az'
                }
        """
        if self.trajectory is None:
            raise RuntimeError("Spline has not been computed. Call compute_piecewise_polynomial()/compute_spline() first.")
        if deltat is None or deltat <= 0:
            raise ValueError("deltat must be a positive number.")

        t0, tf = float(self.times[0]), float(self.times[-1])

        # Time grid including the final instant exactly
        n_steps = int(np.floor((tf - t0) / deltat)) + 1
        ts = t0 + np.arange(n_steps, dtype=float) * deltat
        if ts[-1] < tf:
            ts = np.append(ts, tf)

        # Vectorized evaluation (each returns shape (len(ts),))
        pos = np.vstack([s(ts) for s in self.trajectory])               # (3, N)
        vel = np.vstack([s.derivative(1)(ts) for s in self.trajectory]) # (3, N)
        acc = np.vstack([s.derivative(2)(ts) for s in self.trajectory]) # (3, N)

        return {
            'x': pos[0], 'y': pos[1], 'z': pos[2],
            'vx': vel[0], 'vy': vel[1], 'vz': vel[2],
            'ax': acc[0], 'ay': acc[1], 'az': acc[2],
        }
    
    def compute_piecewise_polynomial(self, order: int = 3):
        """
        Build a piecewise polynomial of arbitrary degree (order) per dimension (x,y,z),
        concatenating multiple polynomials (one per interval), using SciPy's make_interp_spline.
        This function supports *endpoint* derivative constraints via 'add_waypoint_constraint'.
        Interior-knot constraints are not supported by make_interp_spline and are ignored here.
        """
        if len(self.waypoints) < 2:
            raise ValueError("At least two waypoints are required to compute a piecewise polynomial.")
        if order < 1:
            raise ValueError("order must be >= 1.")

        waypoints = np.asarray(self.waypoints, dtype=float)  # (M,3)
        times = np.asarray(self.times, dtype=float)
        M = len(times)

        # Collect endpoint constraints from self.waypoint_constraints:
        # - at waypoint 0, use "right" side (segment 0, tau=0)
        # - at waypoint M-1, use "left" side  (segment M-2, tau=T_{M-2})
        # bc_type per axis will be (left_constraints, right_constraints)
        # where each side is a list of (derivative_order, value) pairs.
        # Only d in [0 .. order-1] is meaningful for a degree-'order' spline.
        def _endpoint_bc_for_axis(dim: int):
            """
            Build a SciPy-compatible bc_type for make_interp_spline.

            Rules:
            - If cubic (order == 3) and BOTH ends specify derivative d=1,
                return 'clamped' (first derivative fixed at both ends).
            - If nothing specified, return 'not-a-knot'.
            - Otherwise, return 'not-a-knot' (safe fallback);
                SciPy rejects many explicit tuples unless they match its exact counting rules.
            """
            M = len(times)
            info0 = []  # constraints at first knot (right side)
            infoN = []  # constraints at last knot (left side)

            # First knot (index 0) -> use right-side values
            for d, entries in self.waypoint_constraints.items():
                if d < 0 or d > order - 1:
                    continue
                ent = entries.get(0, None)
                if ent is not None and ent["side"] in ("right", "both", "auto"):
                    v = float(ent["value"][dim])
                    if not np.isnan(v):
                        info0.append((d, v))

            # Last knot (index M-1) -> use left-side values
            for d, entries in self.waypoint_constraints.items():
                if d < 0 or d > order - 1:
                    continue
                ent = entries.get(M - 1, None)
                if ent is not None and ent["side"] in ("left", "both", "auto"):
                    v = float(ent["value"][dim])
                    if not np.isnan(v):
                        infoN.append((d, v))

            if order == 3:
                has_right_v = any(d == 1 for d, _ in info0)
                has_left_v  = any(d == 1 for d, _ in infoN)
                if has_right_v and has_left_v:
                    return "clamped"

            if not info0 and not infoN:
                return "not-a-knot"

            # Fallback for any other pattern
            return "not-a-knot"


        splines_ppoly = []

        for dim, label in enumerate(["x", "y", "z"]):
            bc_type = _endpoint_bc_for_axis(dim)
            # Build B-spline with chosen degree and endpoint BCs
            bspline = make_interp_spline(times, waypoints[:, dim], k=order, bc_type=bc_type)
            # Convert to power-basis piecewise polynomial
            ppoly = PPoly.from_spline(bspline)
            splines_ppoly.append(ppoly)

            # Optional: pretty print knot-interval coefficients (descending powers as in PPoly)
            print(f"\nPiecewise polynomial coefficients for {label} (degree={order}):")
            for seg in range(ppoly.c.shape[1]):
                coeffs = ppoly.c[:, seg]  # descending powers
                coeff_str = ", ".join([f"a{j}={coeffs[j]:.6f}" for j in range(coeffs.size)])
                print(f"  Knot interval {seg}: {coeff_str}")

        self.trajectory = splines_ppoly

    # def compute_minimum_snap(self, order: int = 7, continuity_order: int = 3, reg: float = 1e-9):
    #     """
    #     Minimum-snap trajectory using normalized local time u = (t - t_s)/T_s ∈ [0,1].
    #     This is the only supported minimum-snap implementation in this class.
    #     """
    #     return self.compute_minimum_snap_normalized(order=order,
    #                                                 continuity_order=continuity_order,
    #                                                 reg=reg)



    
    
    def compute_minimum_snap(self, order: int = 7, continuity_order: int = 3, reg: float = 1e-9):
        """
        Minimum-snap with normalized local time u = (t - t_s)/T_s ∈ [0,1].
        - Cost uses derivative_cost_Q_block_norm (1/T^{2k-1} scaling) with k=4.
        - Constraints (waypoints, continuity, endpoint vel/acc) use deriv_coeffs_row_norm with u ∈ {0,1}.
        """

        def _falling_factorial(i: int, k: int) -> float:
            """i * (i-1) * ... * (i-k+1); equals 0 if k > i."""
            if k > i:
                return 0.0
            prod = 1.0
            for m in range(k):
                prod *= (i - m)
            return prod


        def deriv_coeffs_row_norm(deg: int, d: int, u: float, T: float) -> np.ndarray:
            """
            Row r such that r @ a = d^d/dt^d p(u) with u in [0,1], t = t_s + u*T.
            Chain rule: d/dt = (1/T) d/du  => d^d/dt^d = (1/T^d) d^d/du^d.
            For p(u) = sum_{k=0..deg} a_k u^k:
                d^d/du^d p(u) = sum_{k=d..deg} (k falling d) * a_k * u^{k-d}.
            Therefore:
                r[k] = (k falling d) * u^{k-d} / T^d,   for k >= d
                r[k] = 0                                for k < d
            """
            r = np.zeros(deg + 1, dtype=float)
            Tpow = (T ** d) if d > 0 else 1.0
            for k in range(d, deg + 1):
                r[k] = _falling_factorial(k, d) * (u ** (k - d)) / Tpow
            return r


        def derivative_cost_Q_block_norm(deg: int, T: float, k: int = 4, reg: float = 1e-9) -> np.ndarray:
            """
            Per-segment Q for ∫_0^T (p^{(k)}(t))^2 dt with u = (t - t_s)/T ∈ [0,1].
            Using t = t_s + uT and d^k/dt^k = (1/T^k) d^k/du^k:
                ∫_0^T (p^{(k)}(t))^2 dt = (1/T^{2k-1}) ∫_0^1 (p^{(k)}(u))^2 du
            For p(u) = sum a_i u^i:
                Q_ij = (i^(k) * j^(k)) / ((i + j - 2k + 1) * T^{2k-1}),   for i,j >= k
                    0 otherwise
            where i^(k) = i*(i-1)*...*(i-k+1) (falling factorial).
            """
            Q = np.zeros((deg + 1, deg + 1), dtype=float)
            Tpow = T ** (2 * k - 1)
            for i in range(k, deg + 1):
                ik = _falling_factorial(i, k)
                for j in range(k, deg + 1):
                    jk = _falling_factorial(j, k)
                    denom = (i + j - 2 * k + 1)  # exponent+1 from ∫_0^1 u^{i+j-2k} du
                    Q[i, j] = (ik * jk) / (denom * Tpow)
            # small regularization for numerical stability
            Q += reg * np.eye(deg + 1)
            return Q    


        if len(self.waypoints) < 2:
            raise ValueError("At least two waypoints are required.")
        if order < 4:
            raise ValueError("order must be >= 4 for snap.")
        if continuity_order < 0 or continuity_order > order - 1:
            raise ValueError("continuity_order must be in [0, order-1].")

        waypoints = np.asarray(self.waypoints, dtype=float)  # (M,3)
        times = np.asarray(self.times, dtype=float)          # (M,)
        M = len(times)
        S = M - 1
        if np.any(np.diff(times) <= 0):
            raise ValueError("Times must be strictly increasing.")

        dt = np.diff(times).astype(float)    # segment durations
        deg = order
        ncoef_seg = deg + 1
        ncoef = S * ncoef_seg

        # ---- Build block-diagonal Q (snap, k=4) with normalized time ----
        Q = np.zeros((ncoef, ncoef), dtype=float)
        offset = 0
        for s in range(S):
            Qs = derivative_cost_Q_block_norm(deg, dt[s], k=4, reg=reg)
            Q[offset:offset + ncoef_seg, offset:offset + ncoef_seg] = Qs
            offset += ncoef_seg

        # ---- Build A templates (independent of axis; b differs per axis) ----

        # 1) Waypoint position constraints at u = 0 or 1
        # Convention: waypoint 0 at start of seg 0 (u=0),
        # interior waypoints k=1..M-2 at END of seg (k-1) (u=1),
        # final waypoint at END of last seg (S-1) (u=1).
        A_pos_template = []
        
        # waypoint 0 on seg 0 at u=0
        row = np.zeros(ncoef, dtype=float)
        row[0:ncoef_seg] = deriv_coeffs_row_norm(deg, d=0, u=0.0, T=dt[0])
        A_pos_template.append(row)

        # interior waypoints
        for k in range(1, M - 1):
            s = k - 1
            base = s * ncoef_seg
            row = np.zeros(ncoef, dtype=float)
            row[base:base + ncoef_seg] = deriv_coeffs_row_norm(deg, d=0, u=1.0, T=dt[s])
            A_pos_template.append(row)
        
        # final waypoint at end of last segment
        base = (S - 1) * ncoef_seg
        row = np.zeros(ncoef, dtype=float)
        row[base:base + ncoef_seg] = deriv_coeffs_row_norm(deg, d=0, u=1.0, T=dt[S - 1])
        A_pos_template.append(row)

        # 2) Continuity at internal knots for derivatives d=0..continuity_order:
        # p_s^{(d)}(u=1) == p_{s+1}^{(d)}(u=0)
        A_cont_template = []
        for s in range(S - 1):
            baseL = s * ncoef_seg
            baseR = (s + 1) * ncoef_seg
            for d in range(0, continuity_order + 1):
                row = np.zeros(ncoef, dtype=float)
                # left end u=1 on seg s
                left_vec = deriv_coeffs_row_norm(deg, d=d, u=1.0, T=dt[s])
                # right start u=0 on seg s+1
                right_vec = deriv_coeffs_row_norm(deg, d=d, u=0.0, T=dt[s + 1])
                row[baseL:baseL + ncoef_seg] = left_vec
                row[baseR:baseR + ncoef_seg] -= right_vec
                A_cont_template.append(row)

        def _append_row_for_knot_norm(A_blocks, b_vals, dim, d, k_idx):
            """
            Add one constraint for derivative d at knot k_idx on axis 'dim' using normalized rows.
            """
            info = self.waypoint_constraints.get(d, {}).get(k_idx, None)
            if info is None:
                return
            target = float(info["value"][dim])
            if np.isnan(target):
                return

            side = info["side"]

            def _add_one(which):
                if which == "left":
                    if k_idx == 0:  # nothing to the left of the first knot
                        return
                    s = k_idx - 1
                    u = 1.0
                    base = s * ncoef_seg
                    row = np.zeros(ncoef, dtype=float)
                    row[base:base + ncoef_seg] = deriv_coeffs_row_norm(deg, d=d, u=u, T=dt[s])
                    A_blocks.append(row); b_vals.append(target)
                elif which == "right":
                    if k_idx == M - 1:  # nothing to the right of the last knot
                        return
                    s = k_idx
                    u = 0.0
                    base = s * ncoef_seg
                    row = np.zeros(ncoef, dtype=float)
                    row[base:base + ncoef_seg] = deriv_coeffs_row_norm(deg, d=d, u=u, T=dt[s])
                    A_blocks.append(row); b_vals.append(target)
                else:
                    raise ValueError("side must be 'left' or 'right'.")

            if side == "both":
                _add_one("left"); _add_one("right")
            elif side == "auto":
                _add_one("right" if k_idx == 0 else "left")
            else:
                _add_one(side)


        # ---- Solve per axis (x,y,z) with same A-structure but different b ----
        PP_list = []
        for dim in range(3):
            A_blocks = []
            b_vals = []

            # waypoint positions
            for idx, row in enumerate(A_pos_template):
                A_blocks.append(row)
                b_vals.append(float(waypoints[idx, dim]))

            # continuity rows
            for row in A_cont_template:
                A_blocks.append(row)
                b_vals.append(0.0)

            # (3) General knot constraints (any derivative, any knot)
            for d in sorted(self.waypoint_constraints.keys()):
                for k_idx in sorted(self.waypoint_constraints[d].keys()):
                    _append_row_for_knot_norm(A_blocks, b_vals, dim, d, k_idx)

            A = np.vstack(A_blocks)           # (n_eq, ncoef)
            b = np.asarray(b_vals, float)     # (n_eq,)

            # KKT solve: [Q A^T; A 0][c; λ] = [0; b]
            KKT = np.block([[Q, A.T],
                            [A, np.zeros((A.shape[0], A.shape[0]))]])
            rhs = np.concatenate([np.zeros(ncoef, float), b])
            KKT += reg * np.eye(KKT.shape[0])  # global tiny reg

            sol = np.linalg.solve(KKT, rhs)
            c = sol[:ncoef]
            # res = A @ c - b
            # print("max |A c - b| =", np.max(np.abs(res)))

            # Pack coefficients into PPoly
            Cdim = np.zeros((deg + 1, S), dtype=float)
            for s in range(S):
                # Coefficients in u (what the solver returns): [a0, a1, ..., a_deg]
                block_u = c[s * ncoef_seg:(s + 1) * ncoef_seg]

                # Rescale to τ-basis: a_tau[k] = a_u[k] / T_s^k
                T = dt[s]
                powers = T ** np.arange(0, deg + 1, dtype=float)  # [T^0, T^1, ..., T^deg]
                block_tau = block_u / powers

                # PPoly expects descending power order: [a_deg, ..., a0]
                Cdim[:, s] = block_tau[::-1]

            P = PPoly(c=Cdim, x=times)  # 'times' are the physical breakpoints [t0, t1, ..., tM-1]
            PP_list.append(P)


        # Activate the solution
        self.trajectory = PP_list


