<h1>Manta Ray Surface Simulation</h1>

<p align="center">
  <img src="assets/itu-demo.png" alt="ITU demo — balls guided to a rectangular formation on the fabric surface" width="600">
</p>

<p>
  A physics simulation of balls rolling on a deformable fabric surface suspended by a grid of
  controllable rods. Each rod can be raised or lowered by a controller, creating local slopes
  that guide ball movement across the surface.
</p>

<p>
  The fabric sag between rods is modelled with catenary curves, and ball dynamics
  (contact, collisions, friction, spin) are solved using XPBD
  (Extended Position Based Dynamics).
</p>

<p>
  Movement is directed by a <em>direction map</em> — a 2D grid of compass directions
  (N, S, E, W) that tells each rod which way to push nearby balls.
  Three controller strategies are available:
  <strong>blocking</strong> (waits for a free destination),
  <strong>non-blocking</strong> (pushes regardless), and
  <strong>priority</strong> (tries directions in order of preference).
</p>

<h2>Running</h2>

<p>Run a simulation and save results to a data file:</p>
<pre>
python run.py                                          # default experiment
python run.py --experiment experiments/itu_demo_stable.py     # custom experiment
python run.py --no-save                                # run without saving data
</pre>

<p>Visualize saved results:</p>
<pre>
python visualize.py opengl       # interactive 3D viewer
python visualize.py video        # export MP4
</pre>

<h2>Experiments</h2>

<p>
  Experiments are plain Python files in <code>experiments/</code>.
  Each file defines a <code>DIRECTION_MAP</code> and a controller name — the grid size
  is derived automatically from the map dimensions.
</p>

<table>
  <tr><th>Experiment</th><th>Status</th><th>Notes</th></tr>
  <tr><td><code>experiments/itu_demo_stable.py</code></td><td>Stable</td><td>Balls guided into a rectangular formation (ITU demo, see image above)</td></tr>
  <tr><td><code>experiments/itu_demo_chaotic.py</code></td><td>Stable</td><td>Chaotic controller variant of the ITU demo</td></tr>
  <tr><td><code>experiments/center_demo.py</code></td><td>Stable</td><td>Balls converge toward the center</td></tr>
  <tr><td><code>experiments/sab_demo_preemptive.py</code></td><td>Published</td><td>SAB 2026 — priority-preemptive controller, deadlock avoidance (tag <code>SAB2026</code>, see <code>sab2026/</code>)</td></tr>
</table>

<p>
  Newer controllers and experiments (weighted bin-covering, distributed
  coverage, pairwise-exchange, stochastic self-assembly) are under active
  development on the <code>dev</code> branch and are not yet merged to
  <code>main</code> — they haven't been demoed or validated enough to call
  stable. <code>main</code> only carries controllers/experiments that run
  and have been checked; <code>dev</code> is where new ones are built up
  before merging.
</p>

<p>
  Each paper's exact code is preserved with a git tag (e.g. <code>SAB2026</code>),
  so results stay reproducible even as <code>main</code> keeps evolving:
</p>
<pre>
git checkout SAB2026    # code as submitted for the SAB 2026 paper
</pre>
