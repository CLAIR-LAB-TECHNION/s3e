from semantic_state_estimator.state_estimator import PredFnStateEstimator


class PRBGTStateEstimator(PredFnStateEstimator):
    def get_obj(self, obj_id, state):
        obj = state.get_object_by_name(obj_id)
        assert obj is not None, "called action with non-existing object"
        return obj

    def is_directly_on_table(self, b, state):
        b_obj = self.get_obj(b, state)

        return not state.objects_below(b_obj)

    def is_clear_on_top(self, b, state):
        b_obj = self.get_obj(b, state)
        return b_obj in state.tops()

    def block_on_top_of_block(self, b1, b2, state):
        b1_obj = self.get_obj(b1, state)
        b2_obj = self.get_obj(b2, state)
        return state.object_just_below(b1_obj) == b2_obj
