# Palletizer Trajectory Assets

## Overview

This data is a recorded palletizing cycle for a [Fanuc](https://www.fanuc.com/) M-710iD/50M robot arm with a suction gripper.

The data is in `palletizer_trajectory.csv`. It holds the joint angles and gripper signals over time.

## Format

Each row is one time sample. The columns are:

| Column | Meaning |
|---|---|
| `time_s` | Time in seconds. |
| `J1_deg` to `J6_deg` | Joint angles in degrees. `J3` is relative to `J2`, as in `J3_deg_rel_J2`. |
| `gripper_close` | `1` when the gripper is commanded to close, else `0`. |
| `prepare_to_engage` | `1` while the gripper prepares to engage the box, else `0`. |

## Sources

The trajectories originally came from FANUC. They were stored as a JSONL file. JSONL means JSON Lines: one JSON record per line.

### Changes

The following changes were made to the source data.

- The format was changed from JSONL to CSV. CSV means comma-separated values: one row per time sample.
- The data was reduced to the fields `time_s`, `J1_deg` to `J6_deg` and `gripper_close`.
- The `prepare_to_engage` column was added for the Newton surface gripper example. It was not in the source data.

## Attribution

Original trajectory author: FANUC. Adapted for Newton as described above. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The license text was copied from the [Creative Commons website](https://creativecommons.org/licenses/by/4.0/legalcode.txt).

Fanuc is a trademark of FANUC Corporation. This data is not endorsed by FANUC Corporation.

## License

This data is released under the [Creative Commons Attribution 4.0 International License](LICENSE).
