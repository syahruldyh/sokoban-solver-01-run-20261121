# sokoban-solver-01

A Sokoban solver that answers a whole set of levels in one run.

`main.ts` reads each instance's `level.txt` (XSB) from `/input/<id>/` and
writes the player's moves to `/output/<id>/solution.txt`. The levels of
this competition are a one-square-wide hallway with independent rooms off
it, so the solver splits a level into its rooms, solves each room with a
push-based A\* (states are box squares plus the player's reachable region;
the heuristic is a minimum-cost matching of boxes to goals over
single-box push distances), and stitches the rooms' move strings
together. Pushes that strand a box on a dead square or freeze a 2x2 block
are pruned, and a whole-level search is the fallback when a level is not
a set of rooms.

On the kit's practice set it solves all five levels in the push-optimal
number of pushes.

[make.fun bounty](https://make.fun/bounty/sokoban-solver-01-run-20261121)
