(define (domain item-sorting)
  (:requirements :strips :typing :equality)
  (:types item table)
  (:predicates (on-table ?i - item ?t - table)
	           (robot-gripper-empty)
	           (robot-gripping ?i - item)
 )

  (:action pick-up
	     :parameters (?i - item ?t - table)
	     :precondition (and (on-table ?i ?t)(robot-gripper-empty))
	     :effect
	     (and (not (on-table ?i ?t))
		   (not (robot-gripper-empty))
		   (robot-gripping ?i)))

  (:action put-down
	     :parameters (?i - item ?t - table)
	     :precondition (robot-gripping ?i)
	     :effect
	     (and (not (robot-gripping ?i))
		   (robot-gripper-empty)
		   (on-table ?i ?t)))
)