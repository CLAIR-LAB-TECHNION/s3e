# Instance sets

Generated, not committed: `make instances` writes the rendered Blocksworld set
here (deterministic in its seeds), and `make instances-viplan` exports recorded
ViPlan++ episodes. Each scene is a directory with `domain.pddl`,
`problem.pddl` (true initial state and goal), `state.json` (every grounded
atom's truth value), `meta.json`, and `view<k>.png`; `index.json` records the
generator arguments and provenance.
