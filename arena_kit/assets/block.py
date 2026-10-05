# Default opponent controller: the stationary block.
#
# The block (assets/block.xml) is a heavy box with a freejoint and NO actuators,
# so its controller commands nothing. It never moves — a sparring dummy you must
# actually shove off the ring. You are free to fight any other opponent instead:
# pass --opponent_xml / --opponent_controller (or the run_match tool's
# opponent_xml / opponent_controller args) pointing at any robot you like.


def policy_step(obs):
    del obs  # the block takes no actions
    return {}
