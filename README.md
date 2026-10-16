# causal-rescue-robot
Causal_robot: rescue-robot debris removal with a physics model in the loop

A UR5 with a Robotiq 2F-85 uncovers a target buried in a collapsed pile of debris
inside a tote. At every step the loop observes the pile, builds the legal actions
(picks that pass the geometric gates and blade pushes whose descent is clear),
ranks them with a transformer trained on interventional labels, checks the top
five by applying them in the pile model, vetoes the ones that make the burial
worse, executes the best survivor on the simulated arm and re-observes. The run
ends when the exposure of the target reaches 0.95.

Install

    pip install -r requirements.txt

PyBullet, NumPy, PyTorch, OpenCV, Pillow and ffmpeg are required. The ompl
package gives collision-aware transit planning; without it the controller falls
back to direct moves.

Run the demo

    python3 demo.py --pile aval_het --seed 101
    python3 demo.py --pile aval_hom50 --seed 100

The first command runs the 50-piece mixed-debris collapse of seed 101 (exposure
0.026 to 1.000 in five actions), the second the 50-cube collapse of seed 100
(0.000 to 1.000 in ten actions). Each writes output/demo_<pile>_seed<seed>.mp4
(1280x720, 30 fps, real time while the gripper is in the pile, faster in
transit, a top-down decision frame before every action) together with the step
log run_<pile>_seed<seed>.json and the per-step audit of contacts and forces.
A run takes about 50 to 90 minutes on two cores; most of it is rendering.
With --no-video the same run finishes in about 10 to 30 minutes and only the
logs are written.

Scene presets are in robot_demo/scenes.py: aval_het (50 mixed pieces),
aval_het_tall, aval_het_slide, aval_het80, aval_het80_tall, aval_het80_slide
(80 pieces), aval_hom30, aval_hom40, aval_hom50 (identical cubes). The seed
decides the collapse. Other seeds are not guaranteed to be rescued; the paper
reports the rates.

Layout

    demo.py                          entry point
    sim.py                           scene, arm, gripper, piles
    production_robot_controller.py   grasp pipeline, blade push, planning
    robot_geometry.py                gripper geometry
    rubble_generator.py              mixed debris pieces
    perception/                      exposure, burial depth, occluders
    causal_core/                     gates, action space, features, labels
    decision/                        ranker and its loader
    robot_demo/                      scenes, closed loop, recorder, preflight
    training/                        data collection, relabelling, training
    models/cat_v11/best.pt           the ranker used by the demo
    assets/                          URDF and meshes

Training the ranker again

    python3 training/collect.py --pile aval_het80 --seeds 0-9 --max-steps 30 --cf-settle 120 --cf-settle2 100 --cf2-width 3 --out datasets/het80.csv
    python3 training/relabel.py datasets/het80.csv --promise 1.0 --out datasets/het80_full.csv
    CAT_VAL_FRAC=0.2 python3 training/train.py --csv datasets/het80_full.csv --target_col cf_value2 --access --min-spread 1e-4 --causal-mode all --mediators cf_delta_E,cf_delta_layers,cf_unlock_L,cf_delta_clear,cf_delta_options --mediator-weight 0.3 --save_dir models/cat_new

Collection applies every legal action of every visited state in the pile model
and records the effect; it is slow (hours per ten episodes on two cores). The
shipped checkpoint was trained on 591 states from the 80-piece scenes and the
50-piece base scene.

Checks

    python3 robot_demo/preflight.py
    python3 -m robot_demo.rescue_demo --mode rescue --pile aval_het --seed 101 --policy cat --model models/cat_v11/best.pt --actions pick_push --verify-k 5 --max-steps 24
