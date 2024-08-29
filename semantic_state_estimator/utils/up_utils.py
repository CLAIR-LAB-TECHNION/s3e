from itertools import product

from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.model import Problem


def create_up_problem(domain, problem):
    reader = PDDLReader()
    if domain.lower().endswith(".pddl"):
        assert problem.lower().endswith(".pddl"), "if domain is a file, problem must also be a file"
        up_problem = reader.parse_problem(domain, problem)
    else:
        up_problem = reader.parse_problem_string(domain, problem)

    return up_problem


def get_object_names_dict(up_problem):
    objects = {}
    for t in up_problem.user_types:
        objects[t.name] = list(map(str, up_problem.objects(t)))

    return objects


def get_all_grounded_predicates_for_objects(up_problem, objects=None):
    predicates = up_problem.fluents
    if objects is None:
        objects = get_object_names_dict(up_problem)

    grounded_predicates = []
    for p in predicates:
        varlists = []
        for variable in p.signature:
            varlists.append(objects[variable.type.name])
        for assignment in product(*varlists):
            grounded_predicates.append(f'{p.name}({",".join(assignment)})')

    return grounded_predicates


def get_pddl_files_str(up_problem):
    writer = PDDLWriter(up_problem)
    return writer.get_domain(), writer.get_problem()


def ground_predicate_str_to_fnode(up_problem, predicate_str):
    fluent_name, args = predicate_str.split('(')
    args = args.rstrip(')').split(',')
    args = [arg.strip() for arg in args if arg]
    pred_obj = up_problem.fluent(fluent_name)
    arg_obj = [up_problem.object(a) for a in args]
    if arg_obj:
        return pred_obj(*arg_obj)
    else:
        return pred_obj()


def bool_constant_to_fnode(up_problem: Problem, constant: bool):
    exp_mgr = up_problem.environment.expression_manager
    if constant is True:
        return exp_mgr.true_expression
    else:
        return exp_mgr.false_expression
