README:

This file provides a list of experiments used for testing the algorithms described in [1].

List of experiments:
1) experiment1_lyapunov_sp: standard Lyapunov equation with a sparse banded coefficient matrix, used for quick validation.
2) experiment2_iga_sp: isogeometric analysis on a plate with a hole. Linear system with the mass matrix, recast as 10-term matrix equation with sparse banded coefficient matrices. This experiment requires the GeoPDEs package [2].
3) [To be added]
4) experiment4_circuit_sp: RC circuit model, 3-term Lyapunov-plus-positive matrix equation with sparse full-rank coefficients for the Lyapunov part and low-rank coefficients for the additional term. Example inspired from [3,4].
5) experiment5_heat_sp: Controlled heat equation with Robin boundary conditions, 6-term Lyapunov-plus-positive matrix equation with singular Lyapunov part. Example inspired from [5].
6) [To be added]
7) experiment7_convection_diffusion_sp: finite difference discretization of a convection-diffusion equation, 4-term Sylvester equation based on Example 4 in [6].

Author: Yannis Voet
Last updated: February, 2024
Coded on MATLAB R2023a
 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

References:
[1] Y. Voet. Preconditioning techniques for generalized Sylvester matrix equations
[2] R. Vázquez. A new design for the implementation of isogeometric analysis in Octave and Matlab: Geopdes 3.0. Computers & Mathematics with Applications, 72(3):523–554, 2016.
[3] Z. Bai and D. Skoogh. A projection method for model reduction of bilinear dynamical systems. Linear algebra and its applications, 415(2-3):406–425, 2006.
[4] P. Benner and T. Breiten. Low rank methods for a class of generalized Lyapunov equations and related issues. Numerische Mathematik, 124(3):441–470, 2013.
[5] T. Damm. Direct methods and ADI-preconditioned Krylov subspace methods for generalized Lyapunov equations. Numerical Linear Algebra with Applications, 15(9):853–871, 2008.
[6] D. Palitta and V. Simoncini. Matrix-equation-based strategies for convection–diffusion equations. BIT Numerical Mathematics, 56:751–776, 2016.
