"""Benchmark-definition numerics for the paper's beam cases (NumPy/SciPy only, float64)."""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

E, rho, a = 2.0e11, 7845.0, 0.030
A, I = a*a, a**4/12
EI, rhoA = E*I, rho*A
c2 = EI/rhoA
print(f"EI={EI:.4f}  rhoA={rhoA:.6f}  c^2=EI/rhoA={c2:.4f}  c={np.sqrt(c2):.5f}  paper c^2=43.73^2={43.73**2:.4f}")

roots = {
 "fixed-fixed":  brentq(lambda z: np.cos(z)*np.cosh(z)-1, 4.0, 5.0),
 "simply-supp":  np.pi,
 "cantilever":   brentq(lambda z: np.cos(z)*np.cosh(z)+1, 1.5, 2.5),
}
paper_bl = {"fixed-fixed": 4.7300, "simply-supp": 3.1416, "cantilever": 1.8751}
paper_f  = {"fixed-fixed": 20.594, "simply-supp": 9.085, "cantilever": 1.529}
Ls = {"fixed-fixed": 2.75, "simply-supp": 2.75, "cantilever": 4.0}
for k, bl in roots.items():
    L = Ls[k]; w = (bl/L)**2*np.sqrt(c2); f = w/2/np.pi
    print(f"{k:12s} beta1*l exact={bl:.9f} paper={paper_bl[k]}  w1={w:.4f} rad/s  f1={f:.4f} Hz (paper {paper_f[k]})  cycles in [0,1]s={f:.2f}")
    for b in (0.0, 50.0, 5.0):
        g = b/rhoA
        print(f"      b={b:5.1f} gamma=b/rhoA={g:.4f}  zeta=gamma/(2w)={g/(2*w):.5f}  envelope at t=1: {np.exp(-g/2):.4f}")

def q(t, w, g):
    wd = np.sqrt(w*w - g*g/4); return np.exp(-g*t/2)*(np.cos(wd*t) + g/(2*wd)*np.sin(wd*t))

print("\nL2 FLOOR from a reference/PDE frequency mismatch (rel-L2 between q(w) and q(w(1+d)) on t in [0,T]):")
t = np.linspace(0, 1, 20001)
for case, w, g, T in [("FE undamped", (roots['fixed-fixed']/2.75)**2*np.sqrt(c2), 0.0, 1),
                      ("FE damped b=50", (roots['fixed-fixed']/2.75)**2*np.sqrt(c2), 50/rhoA, 1),
                      ("SS undamped", (np.pi/2.75)**2*np.sqrt(c2), 0.0, 1),
                      ("SS damped b=50", (np.pi/2.75)**2*np.sqrt(c2), 50/rhoA, 1)]:
    q0 = q(t, w, g)
    for label, d in [("beta1l 4.7300 vs exact", 2*(4.7300-roots['fixed-fixed'])/roots['fixed-fixed']),
                     ("c^2 = 43.73^2 vs EI/rhoA", 0.5*(43.73**2-c2)/c2),
                     ("paper f1 table vs computed", None)]:
        if label.startswith("beta") and not case.startswith("FE"): continue
        if d is None:
            d = (20.594/20.5886 - 1) if case.startswith("FE") else (9.085/9.0825 - 1)
        e = np.linalg.norm(q(t, w*(1+d), g) - q0)/np.linalg.norm(q0)
        print(f"   {case:15s} {label:28s} d={d:+.3e}  -> rel-L2 floor {e:.2e}")

print("\nModal content of the paper's static-deflection ICs (fraction of ||u0||_2 NOT in mode 1):")
L = 2.75
bl = roots["fixed-fixed"]; be = bl/L
sg = (np.cosh(bl)-np.cos(bl))/(np.sinh(bl)-np.sin(bl))
U1 = lambda x: np.cosh(be*x)-np.cos(be*x)-sg*(np.sinh(be*x)-np.sin(be*x))
u0h = lambda x: x**2*(4*x-3*L)          # Eq. 27 shape (F/48EI dropped), x <= L/2
u0 = lambda x: u0h(x) if x <= L/2 else u0h(L-x)
p = quad(lambda x: u0(x)*U1(x), 0, L, limit=200)[0]/quad(lambda x: U1(x)**2, 0, L)[0]
res = quad(lambda x: (u0(x)-p*U1(x))**2, 0, L, limit=200)[0]/quad(lambda x: u0(x)**2, 0, L, limit=200)[0]
print(f"   fixed-fixed Eq.27: residual fraction {np.sqrt(res):.3e};  mode-1 amplitude ratio mid-span = {p*U1(L/2)/u0(L/2):.5f}")
u0s = lambda x: (x*(4*x*x-3*L*L)) if x <= L/2 else ((L-x)*(4*(L-x)**2-3*L*L))
U1s = lambda x: np.sin(np.pi*x/L)
p = quad(lambda x: u0s(x)*U1s(x), 0, L, limit=200)[0]/(L/2)
res = quad(lambda x: (u0s(x)-p*U1s(x))**2, 0, L, limit=200)[0]/quad(lambda x: u0s(x)**2, 0, L, limit=200)[0]
print(f"   simply-supp A.2: residual fraction {np.sqrt(res):.3e}")
print(f"\n   FE mode shape float32 cancellation: cosh(beta*L)={np.cosh(bl):.2f}; |U1|max={np.abs(U1(np.linspace(0,L,1001))).max():.4f}")
x = np.linspace(0, L, 1001).astype(np.float32); be32 = np.float32(be); sg32 = np.float32(sg)
U32 = np.cosh(be32*x)-np.cos(be32*x)-sg32*(np.sinh(be32*x)-np.sin(be32*x))
print(f"   max |U1_float32 - U1_float64| / |U1|max = {np.abs(U32-U1(x.astype(np.float64))).max()/np.abs(U1(x.astype(np.float64))).max():.2e}  (at x=L should be 0)")
print(f"   U1_float32(L) = {U32[-1]:.3e}")
