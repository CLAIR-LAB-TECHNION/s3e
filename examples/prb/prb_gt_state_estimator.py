from semantic_state_estimator.state_estimator import PredFnStateEstimator


class PRBGTStateEstimator(PredFnStateEstimator):
    def get_obj(self, obj_id, state):
        obj = state.get_object_by_name(obj_id)
        assert obj is not None, "called action with non-existing object"
        return obj

    def is_directly_on_table(self, s, state):
        s_obj = self.get_obj(s, state)
        return not state.objects_below(s_obj)

    def is_clear_on_top(self, s, state):
        s_obj = self.get_obj(s, state)
        return s_obj in state.tops()

    def shape_on_top_of_block(self, s, b, state):
        s_obj = self.get_obj(s, state)
        b_obj = self.get_obj(b, state)
        return state.object_just_below(s_obj) == b_obj
