(define (domain item-sorting)
  (:requirements :strips :typing :equality)
  (:types item table container)
  (:predicates (on-table ?i - item ?t - table)
	           (robot-gripper-empty)
	           (robot-gripping ?i - item)
               (in-container ?i - item ?c - container)
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

  (:action put-in-container
	     :parameters (?i - item ?c - container)
	     :precondition (and (robot-gripping ?i))
	     :effect
	     (and (not (robot-gripping ?i))
		   (robot-gripper-empty)
		   (in-container ?i ?c)))

  (:action take-from-container
	     :parameters (?i - item ?c - container)
	     :precondition (and (in-container ?i ?c)(robot-gripper-empty))
	     :effect
	     (and (robot-gripping ?i)
		   (not (robot-gripper-empty))
		   (not (in-container ?i ?c)))))