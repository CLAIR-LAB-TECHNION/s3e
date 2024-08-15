from itertools import product
import tempfile
import os

from unified_planning.io import PDDLReader, PDDLWriter


def create_up_problem(domain, problem):
    if domain.lower().endswith(".pddl"):
        assert problem.lower().endswith(".pddl"), "if domain is a file, problem must also be a file"
        up_problem = create_up_problem_from_pddl_files(domain, problem)
    else:
        up_problem = create_up_problem_from_ppdl_str(domain, problem)
    
    return up_problem


def create_up_problem_from_pddl_files(domain_filename, problem_filename):
    reader = PDDLReader()
    return reader.parse_problem(domain_filename, problem_filename)

def create_up_problem_from_ppdl_str(domain_str, problem_str):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f_dom:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f_prob:
            f_dom.write(domain_str)
            f_prob.write(problem_str)

            f_dom.close()
            f_prob.close()
        
            up_problem = create_up_problem_from_pddl_files(f_dom.name, f_prob.name)
            
            os.remove(f_dom.name)
            os.remove(f_prob.name)

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
