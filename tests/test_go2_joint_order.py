from __future__ import annotations

import unittest

from go2_joint_order import GO2_JOINT_ORDER, UNITREE_GO2_DOF_ORDER, build_reorder_indices


class Go2JointOrderTests(unittest.TestCase):
    def test_joint_orders_have_same_names(self) -> None:
        self.assertEqual(len(GO2_JOINT_ORDER), 12)
        self.assertEqual(len(UNITREE_GO2_DOF_ORDER), 12)
        self.assertEqual(set(GO2_JOINT_ORDER), set(UNITREE_GO2_DOF_ORDER))

    def test_asset_to_policy_indices_match_expected_go2_layout(self) -> None:
        indices = build_reorder_indices(UNITREE_GO2_DOF_ORDER, GO2_JOINT_ORDER)
        self.assertEqual(indices, (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8))

    def test_policy_to_asset_round_trip(self) -> None:
        asset_to_policy = build_reorder_indices(UNITREE_GO2_DOF_ORDER, GO2_JOINT_ORDER)
        policy_to_asset = build_reorder_indices(GO2_JOINT_ORDER, UNITREE_GO2_DOF_ORDER)

        asset_values = list(range(12))
        policy_values = [asset_values[index] for index in asset_to_policy]
        restored_asset_values = [policy_values[index] for index in policy_to_asset]

        self.assertEqual(restored_asset_values, asset_values)


if __name__ == "__main__":
    unittest.main()
