Vector Edge Solitons and Domain Walls in a Nonlinear Mechanical Topological Insulator

David D. J. M. Snee and Yi-Ping Ma Department of Mathematics, Physics and Electrical Engineering, Northumbria University, Newcastle upon Tyne, NE1 8ST, UK

We report nonlinear edge waves in a 2D mechanical topological insulator. A bulk lattice consists of pendulums with on-site cubic nonlinearity connected by linear springs realizing quantum spin Hall effect. We show that the nonlinear interaction between two edge modes with equal group velocities (EGV) is described by a 1D two-component coupled nonlinear Schr¨odinger (CNLS) equation. On the interface separating two bulk lattices with opposite spin Chern numbers, we construct linear springs such that the dispersion relation exhibits EGV points with favorable CNLS coefficients. Thus, we realize nonlinear edge waves propagating along the interface, including bright-bright (BB) edge solitons for focusing CNLS coefficients, and dark-dark edge solitons, edge domain walls, and dark-bright edge solitons for defocusing CNLS coefficients. In terms of the site amplitudes, these solutions resemble bright and dark breathers. These solutions should be topologically protected when both carrier frequencies lie within a band gap, which we explicitly show by passing BB edge solitons through compact defects on the interface. We also show energy transfer in BB edge soliton collisions with potential application to collision-based computing. Generally, vector edge solitons exhibit a large parameter space for soliton collisions, which endows mechanical devices with greater potential for information processing and other functionalities.

**I.** **INTRODUCTION**

Topological insulators (TIs) are materials which boast the unique property of conduction on the edge (surface) whilst the bulk remains insulating [1, 2]. Although the 1D Su-Schrieffer-Heeger model is often deemed the sim-plest TI nowadays, the concept of TIs first entered main-stream condensed matter physics via 2D models. The integer quantum Hall (QH) effect was first observed in a landmark experiment of a 2D electron gas subject to a perpendicular magnetic field [3]. Thouless, Kohmoto, Nightingale and den Nijs then showed theoretically that the quantized Hall conductance of the QH state results directly from an integer topological invariant of the bulk band structure called the Chern number [4]. The study of such TIs (Chern insulators) was initiated by Haldane, who showed that broken time-reversal symmetry is re-sponsible for nontrivial bulk topology and chiral edge states [5]. When two materials with different bulk Chern numbers touch each other, the bulk-edge correspondence guarantees the existence of unidirectional edge modes at the interface between the two materials that are immune to backscattering in the presence of weak disorder [6]. Note that one of the two materials can be the vacuum whose Chern number is zero. Later, Kane and Mele found 2D topological phases with time-reversal symmetry preserved [7]. Such models are called quantum spin Hall (QSH) effect since they use two copies of a Chern insulator with opposite spins [8]. The Chern number of either component is proportional to the spin and is often called the spin Chern number. When the spin is conserved, the two components behave

[yiping.ma@northumbria.ac.uk](mailto:yiping.ma@northumbria.ac.uk)

independently and host the same number of edge modes propagating in opposite directions; such edge modes are called helical rather than chiral. More generally, one can allow spin-flips while preserving time-reversal symmetry. Such systems are called QSH insulators or Z TIs since2 they are characterized by a Z topological invariant that2 is the spin Chern number of either component modulo 2 [9, 10]. At the interface between two such materials with different spin Chern numbers, the bulk-edge correspon-dence guarantees the existence of helical edge states that are immune to backscattering in the presence of weak time-reversal-symmetric disorder [11]. The first realistic QSH material was predicted theoretically by Bernevig, Hughes, and Zhang [12] and soon confirmed experimen-tally [13], which established TIs as a major research field. Recently, the theoretical framework of quantum TIs was extended to photonic (electromagnetic) [14], cold atomic [15], and phononic (mechanical) [16] systems. Such extensions of topological phases from quantum to classical waves were initiated by Haldane and Raghu, who proposed a photonic QH analog using a 2D lattice of time-reversal symmetry-breaking elements [17]. Their idea of using magneto-optic effect was realized experi-mentally in the microwave domain [18], but alternative approaches were needed in the optical domain. A pio-neering experiment uses Floquet engineering to realize the Haldane model for QH effect, marking the birth of photonic Floquet TIs [19]. Meanwhile, pioneering work on photonic QSH analogs constructs pseudo-spins in an artificial magnetic field on various platforms including bianisotropic metamaterials [20] and coupled silicon ring resonators [21]. Since these early discoveries, the research field of topological photonics has grown rapidly [14]. In mechanical systems, topological edge modes can be realized at either zero frequency or high frequencies [22]. Here we focus on high-frequency topological mechanics,


---

which permits topological phonon transport [16] and thus parallels topological photonics. In earliest proposals of phononic QH analogs, time-reversal symmetry breaking is first conjectured for microtubules in biophysics [23], but it is first realized in optomechanical crystals [24]. Pioneering work on phononic Chern insulators achieves intrinsic time-reversal symmetry breaking using coupled gyroscopes [25, 26] or Coriolis force [27, 28]. Meanwhile, a pioneering experiment uses coupled pendula to realize a mechanical QSH analog, marking the birth of mechanical TIs (MTIs) [29]. Topological phonon transport is actively studied in not only discrete systems as reviewed above, but also acoustic systems (gases or liquids) and elastic systems (solids) on many length scales [16]. Although classical TIs including photonic, cold atomic, and mechanical TIs resemble quantum TIs linearly, the former can also exhibit rich nonlinear phenomena absent in the latter. The emerging field of nonlinear topological photonics combines topology and nonlinearity to achieve advanced functionalities in optical devices [30–32]. A salient feature of nonlinear TIs is the possible existence of localized states on the edge called edge solitons. In 2D TIs, an edge soliton with a narrow spectral (wide spa-tial) envelope along the edge is topologically protected if its carrier frequency lies within the topological band gap. Such continuous edge solitons are extensively studied in photonic Floquet TIs [33–42] and polariton TIs [43–46]. Meanwhile, discrete edge solitons in photonic Floquet TIs have also attracted great interest [47]. For bright edge solitons, there are typically a two-parameter continuous family with low amplitudes and a one-parameter discrete family with arbitrary amplitudes. Such scalar edge soli-tons in photonic TIs are recently observed in pioneering experiments [48–51], paving the way for device applica-tions. Meanwhile, vector edge solitons in photonic or cold atomic TIs are also found on the interface between a TI and its partner [52], the edge of a TI with two band gaps [53], or a combination thereof [54]. Generally, vector solitons have a larger parameter space and thus exhibit richer dynamics than their scalar counterparts. By contrast, nonlinear topological phononics is still in its early stages, but some central themes have emerged. In Ref. [55], amplitude-dependent edge modes are found in 1D and 2D MTIs with inter-site cubic nonlinearity. In the 2D case, the MTI studied is a bi-layered lattice real-izing a mechanical QSH analog [56], and high-amplitude edge waves are found numerically. In Ref. [57], scalar edge solitons are found analytically in a 2D MTI with on-site cubic nonlinearity that describes the first experi-mental realization of a mechanical QSH analog [29]. Such edge solitons are low-amplitude edge waves preserving their shapes during propagation and thus parallel those in 2D nonlinear photonic or cold atomic TIs. Meanwhile, there are proposals to create topologically protected edge channels using nonlinearity in a 1D lattice with inter-site cubic nonlinearity with alternating signs [58] or a 2D lat-tice with on-site cubic nonlinearity via zone folding [59]. Recently, the existence, stability, and dynamics of nonlin-

2

ear topological edge states are extensively studied in 1D MTIs with on-site cubic nonlinearity [60, 61], inter-site cubic nonlinearity [62, 63] possibly with local resonators [64], or on-site sine nonlinearity [65]. However, nonlinear topological edge states in 2D MTIs are rarely explored to our knowledge. Specifically, although scalar edge solitons are reported [57], vector edge solitons remain open. In this paper, we construct a 2D nonlinear MTI that exhibits vector edge solitons and domain walls (DWs). In Section II, we review the design of a 2D MTI that is a mechanical QSH analog. In Sections III, we derive a 1D coupled nonlinear Schr¨odinger (CNLS) equation from a 2D MTI with on-site cubic nonlinearity and review key solutions of the CNLS equation including vector solitons and DWs. In Sections IV & V, we first design a 2D non-linear MTI with two topological sectors that provides a flexible platform to access different regimes of the 1D CNLS equation. Then, we study numerically the propa-gation of vector edge solitons and DWs in this system. In Sections VI & VII, we further explore bright-bright (BB) edge solitons including their robust propagation around compact defects and their collision properties with pos-sible applications to computing. The paper concludes in Section VIII with some directions for future research.

**II.** **MECHANICAL** **TOPOLOGICAL** **INSULATOR** **AND** **DISPERSION** **RELATION**

Our 2D MTI is an adaptation of the first mechanical analogue of a QSH insulator realized experimentally in Ref. [29]. The mechanical lattice consists of a collection of pendula connected by linear springs. The QSH effect uses two copies of the Hofstadter model [66] on a square lattice indexed by (r, s). Thus, the Hamiltonian is X ˆ

ˆ

H

Hα α

where the Hamiltonian for each spin is given by X ˆ

s

H f â â

eiαΦ â â + H c

α 0 r,s,α r,s+1,α

r,s,α r+1,s,α r,s

Here, α is the spin-index, f0 is the hopping amplitude, âr,s,α and ˆar,s,α are respectively the creation and annihi-lation operators of a particle with spin α at site (r, s), Φ is the magnetic flux, and H.c. denotes Hermitian conju-ˆ

gacy. The choice Φ = 2π/3 makes H periodic on a 1 3 unit cell, so the Hamiltonian matrix is

H 0 H (1) 0 H

where H are 3 3 matrices. The key insight of Ref. [29] is that the Hamiltonian matrix H can be made real symmetric via a similar-ity transform. The resulting matrix can then be made positive-definite and serve as the dynamical matrix for


---

coupled oscillators. In the experimental setup, each site of the square lattice hosts two 1D pendula ( both of which swing only in the connections between neighboring sites are realized using springs possibly with lever arms. Hereafter, we group the lattice sites into unit cells indexed by ( consisting of 3 sites ( To enable nonlinear waves in this MTI, we account for the inherent cubic (Duffing) nonlinearity of the pendula. The equations of motion for the 6 pendula ( j = 0 1 2, in the unit cell (

(0) x t) =r,S

f (0) y t) =r,S

f (1) x t) =r,S

f (1) y t) =r,S

f (2) x t) =r,S

f (2) y t) =r,S

f

Here, t denotes time, scribes the linear restoring forces of the springs, and the pendula are assumed identical with angular frequency Consistent with Refs. [29, 57], we choose ω = 3π/2. The nonlinear coefficient must be

0 to yield a cubic approximation to the sinusoidal restor-ing force of a pendulum. The self-coupling coefficient depends on the detailed setup as explained next. For a simple spring connecting a pendulum and its neighbor, the restoring force, which is proportional to the relative displacement, yields a positive coupling to the neighbor and a negative self-coupling. As shown in the first two terms of the second rows in Eqs. (2–7), all couplings to the neighbor in the and thus can be realized by simple springs; see Fig. 1(a).

x yr,s r,s s direction, and the linear

r, S) with each cell j j x y ) for j = 0 1 2.

r,S r,S

x j y j r, S) are written explicitly as (0) (0) ω2 A f x σ x 3

s 0

r,S r,S (1) (2) (0) (0) x x

x

x (2)

r,S r,S 1 r+1,S r 1,S (0) (0) ω2 A f y σ y 3

s 0

r,S r,S (1) (2) (0) (0) y y

y

y (3)

r,S r,S 1 r+1,S r 1,S (1) (1) ω2 A f x σ x 3

s 0

r,S r,S f (1) (1) (0) (2) x

x

x x

r+1,S r 1,S

r,S r,S 2 3f (1) (1) y

y (4)

r+1,S r 1,S 2

(1) (1) A f y σ y 3

ω2 s 0

r,S r,S f (0) (2)

(1) (1) y y

y

y

r,S r,S

r+1,S r 1,S 2 3f (1) (1) x

x (5)

r+1,S r 1,S 2

(2) (2) A f x σ x 3

ω2 s 0

r,S r,S f (0) (1)

(2) (2) x

x

x

x

r,S+1 r,S

r+1,S r 1,S 2 3f (2) (2) y

y (6)

r+1,S r 1,S 2

(2) (2) A f y σ y 3

ω2 s 0

r,S r,S f (0) (1)

(2) (2) y

y

y

y

r,S+1 r,S

r+1,S r 1,S 2 3f (2) (2) x

x (7)

r+1,S r 1,S 2

denotes time derivative, f de-

ω0 f = 4 16π2 and σ ω20

A

s-direction are f > 0 3

**(a)**

** x-pendulum**

**s coupling**

** y-pendulum**

**(b)**

**r coupling**

**positive**

** negative**

**(c)**

**r cross-** **coupling** **left sub-lattice right sub-lattice**

FIG. 1. (Color online) Schematic view of the connections be-tween neighboring pendula in the 2D MTI. The r-direction is horizontal, the s-direction is vertical, and all pendula swing in the s-direction. (a) Simple springs connecting x x and y y pendula in the s-direction. (b) Complex springs connecting x x and y y pendula in the r-direction. (c) Complex springs connecting x y pendula in the r-direction. Here, a complex spring with two springs and one lever arm realizes a nega-tive coupling, while a complex spring with three springs and two lever arms realizes a positive coupling; see Ref. [29] for a physical depiction. If the magnetic flux Φ changes sign, then the r cross-couplings in panel (c) change sign. As explained in Section IV, these two types of r cross-couplings define the two sub-lattices of the 2D MTI with two topological sectors.

In contrast, any connection in the r-direction cannot be realized by a simple spring since the displacements are in the s-direction. Thus, lever arm(s) must be used to rotate the displacements around pivot(s). As shown in the last two terms of the second rows in Eqs. (2–7), the couplings to the neighbor in the r-direction are f for (x(0) y(0)) and f cos Φ = f/2 for (x(1) y(1)) and x(2) y(2) The former requires two lever arms, while the latter requires one lever arm; see Fig. 1(b). The above two types of couplings, i.e., s couplings and r couplings, connect either x x or y y pendula. Mean-while, as shown in the third rows in Eqs. (4–7), a third type of couplings connect x y pendula in the r-direction for (x(1) y(1)) and (x(2) y(2) These r cross-couplings are f sin Φ = 3f/2 times alternating signs, so they change sign when Φ changes sign; see Fig. 1(c). Besides a coupling to the neighbor, either a simple 6 spring or a complex spring with lever arm(s) yields a negative self-coupling. Thus, the total self-coupling for x(1) y(1)) and (x(2) y(2)) is A f where A = 3 + 3.

s

s

s The total self-coupling for (x(0) y(0)) is 4f but can be made A f using springs attached to walls.s For ease of computation, the nonlinear equations of motion (2–7) may be written in the compact matrix form

**X** t) = (LX σN (8)

r,S

r,S r,S

where **X** = [x(0) y(0) x(1) y(1) x(2) y(2) T L is the ma-


---

FIG. 2. (Color online) The dispersion relation α k) obtained from numerical solutions of the eigenvalue problem in Eq. (9). The bulk and edge spectra are shown in blue and red respec-tively. This dispersion relation does not change when the magnetic flux Φ changes sign.

trix encoding the linear couplings, and N **X**³ is the cubic nonlinearity. In the linear limit σ = 0, this system fits into the classification scheme of topological phonons [67]. Since velocity-dependent forces are absent, this sys-tem belongs to the category of reciprocal metamaterials. To obtain all possible symmetries of this system, we study the dynamical matrix D **k**) with 2D wavevector **k**, which is the 2D Fourier transform of L. As shown in Ref. [67], the structure of D **k**) exhibits a T symmetry that squares to +1. Moreover, this symmetry can be augmented to a T symmetry that squares to 1. Here, the T symmetry generalizes the time-reversal symmetry in quantum TIs but does not imply the reversal of time in the mechanical setting. Overall, this system belongs to symmetry class AII in 2D with the presence of the T and T symmetries and the absence of other symmetries, and the bulk topological index can be shown to be Z ; see2 Ref. [67] for detailed analyses of this and related systems. Due to the bulk-edge correspondence, the nontrivial topology of the bulk band structure guarantees the ex-istence of topologically protected helical edge states at the interface between this 2D MTI and the vacuum. To find the dispersion relation of edge states along any direction, say S, we consider the linear problem, i.e., Eq. (8) with σ = 0, and apply the 1D Fourier transform tα k

**X** ei Sk **X** c.c. where k is the wavenum-

r,S

r ber in S α k) is the dispersion relation, and c.c. denotes complex conjugate. This yields the eigenvalue problem

L k **X** α k 2**X**

(9)

r

r

where L k) denotes the matrix L in Eq. (8) after the 1D Fourier transform, and the eigenvector **X** is normalizedr

P **X** 2 = 1.

such that **X** 2

r j

r 2 j The dispersion relation α k k [0 2π), can be nu-merically computed on a finite 1D domain in r. Figure 2 shows this dispersion relation for N = 30 sites withr the bulk (continuous) spectrum shown in blue and the edge (discrete) spectrum shown in red. The T symmetry implies that the band structures, both bulk and edge, are 4

symmetric with respect to k π [67]. This generalized Kramers’ theorem implies that any eigenvector at (k, α has a Kramers partner at (2π k, α), and together they form a Kramers pair. Since the unit cell has three sites, there are three bulk bands and two band gaps. For any α in a band gap, there is a Kramers pair of topologically protected edge states. This pair of edge states are called helical since they have opposite group velocities.

**III.** **DERIVATION** **AND** **SOLUTIONS** **OF** **COUPLED** **NONLINEAR** **SCHRODINGER** **EQUATION**

Consider a generic 2D nonlinear MTI with two branches of the dispersion relation, α k) and β k). To find weakly nonlinear solutions, we let the multiple scale ansatz be a linear superposition of two edge modes:

tα0 0

**X** t) = ϵ A S, τ ei Sk **X**(1)

r,S

r

tβ0 0

B S, τ ei Sk **X**(2) c.c. O ϵ2

r (10)

where the small parameter 0 < ϵ 1 is the amplitude, k is the carrier wavenumber, and α α k ) and β

⁰ 0 0

0 β k ) are the two carrier frequencies. The two edge states0 **X** are eigenvectors of L k ) and normalized such that0

r **X**(1) 2 **X**(2) 2 = 1. The spectral envelope width

r

r 2

2 is assumed to be ϵ, so the two scalar envelopes A and B depend on the slow space variable S ϵ S V t), whereg V is the group velocity. Moreover, both envelopes areg assumed to evolve in the slow time variable τ ϵ²t Substituting the ansatz (10) into Eq. (8) and expand-ing in powers of ϵ, the O ϵ) and O ϵ2) equations are trivial when V α β , where α α k ) and

g 0

0 0

0 β β k Generically, this equal group velocities 0

0 (EGV) condition α β defines a discrete set of k ’s. At0

0 0 O ϵ3), one takes the inner product of the A equation with **X**⁽¹⁾ and the inner product of the B equation with **X**(2)

r

r

P with the inner product defined as **g** **h**

g h , tojjj yield the 1D CNLS equation:

α A + 3A σ|A|2 + 2˜σ|B|2) = 0

iA 0 ˜ ˜

1

2

τ

SS2

(11) β iB 0 B + 3B σ|B|2 + 2˜σ|A|2) = 0

˜ ˜

τ 3

4

SS2

where α α k β β k ), and 0

0

0

0 σ

σ σ

**X**(1) 4 σ

**X**⁽¹⁾**X**(2) 2 1

2

r 4

r r 2 2α

2α

0

0 σ

σ **X**(2) 4 σ

**X**⁽¹⁾**X**(2) 2

σ 4

3

r 4

r r 2 2β

2β

0

0

Note that setting A = 0 or B = 0 in Eq. (11) recovers the scalar nonlinear Schr¨odinger (NLS) equation in Ref. [57]. The two-component CNLS equation is the universal envelope equation for nonlinear interactions between two quasi-monochromatic plane waves [68, 69]. Over decades,
