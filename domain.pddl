(define (domain item-sorting)
  (:requirements :strips :typing :equality)
  (:types item container)
  (:predicates (on-table ?i - item)
	           (gripper-empty)
	           (gripping ?i - item)
               (in-container ?i - item ?c - container)
 )

  (:action pick-up
	     :parameters (?i - item)
	     :precondition (and (on-table ?i)(gripper-empty))
	     :effect
	     (and (not (on-table ?i))
		   (not (gripper-empty))
		   (gripping ?i)))

  (:action put-down
	     :parameters (?i - item)
	     :precondition (gripping ?i)
	     :effect
	     (and (not (gripping ?i))
		   (gripper-empty)
		   (on-table ?i)))

  (:action put-in-container
	     :parameters (?i - item ?c - container)
	     :precondition (and (gripping ?i))
	     :effect
	     (and (not (gripping ?i))
		   (gripper-empty)
		   (in-container ?i ?c)))

  (:action take-from-container
	     :parameters (?i - item ?c - container)
	     :precondition (and (in-container ?i ?c)(gripper-empty))
	     :effect
	     (and (gripping ?i)
		   (not (gripper-empty))
		   (not (in-container ?i ?c)))))