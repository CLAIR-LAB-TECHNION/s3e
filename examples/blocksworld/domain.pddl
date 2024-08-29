(define (domain n-table-blocks-world)
  (:requirements :strips :typing :equality)
  (:types block table)
  (:predicates (on-top-of ?top ?bottom - block)
	           (on-table ?b - block ?t -table)
	           (clear-on-top ?b - block)
	           (robot-gripper-empty)
	           (robot-gripping ?b - block)
 )

  (:action pick-up
	     :parameters (?b - block ?t - table)
	     :precondition (and (clear-on-top ?b)(on-table ?b ?t)(robot-gripper-empty))
	     :effect
	     (and (not (on-table ?b ?t))
		   (not (clear-on-top ?b))
		   (not (robot-gripper-empty))
		   (robot-gripping ?b)))

  (:action put-down
	     :parameters (?b - block ?t - table)
	     :precondition (robot-gripping ?b)
	     :effect
	     (and (not (robot-gripping ?b))
		   (clear-on-top ?b)
		   (robot-gripper-empty)
		   (on-table ?b ?t)))

  (:action stack
	     :parameters (?b1 ?b2 - block)
	     :precondition (and (robot-gripping ?b1) (clear-on-top ?b2))
	     :effect
	     (and (not (robot-gripping ?b1))
		   (not (clear-on-top ?b2))
		   (clear-on-top ?b1)
		   (robot-gripper-empty)
		   (on-top-of ?b1 ?b2)))

  (:action unstack
	     :parameters (?b1 ?b2 - block)
	     :precondition (and (on-top-of ?b1 ?b2) (clear-on-top ?b1)(robot-gripper-empty))
	     :effect
	     (and (robot-gripping ?b1)
		   (clear-on-top ?b2)
		   (not (clear-on-top ?b1))
		   (not (robot-gripper-empty))
		   (not (on-top-of ?b1 ?b2)))))