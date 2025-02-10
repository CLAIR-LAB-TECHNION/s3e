(define (domain item-sorting)
    (:requirements :strips :typing)
    (:types
        drink
        receptical
    )
    (:predicates
        (robot-gripper-empty)
        (robot-holding-drink-in-air ?i - drink)
        (robot-holding-receptical-in-air ?i - receptical)
        (is-full ?i - receptical)
    )

    (:action pick-up-drink
        :parameters (?i - drink)
        :precondition (robot-gripper-empty)
        :effect (and 
            (not (robot-gripper-empty))
            (robot-holding-drink-in-air ?i))
    )

    (:action put-down-dirnk
        :parameters (?i - drink)
        :precondition (robot-holding-drink-in-air ?i)
        :effect (and (not (robot-holding-drink-in-air ?i))
            (robot-gripper-empty))
    )

    (:action pick-up-receptical
        :parameters (?i - receptical)
        :precondition (robot-gripper-empty)
        :effect (and 
            (not (robot-gripper-empty))
            (robot-holding-receptical-in-air ?i)
        )
    )

    (:action put-down-receptical
        :parameters (?i - receptical)
        :precondition (robot-holding-receptical-in-air ?i)
        :effect (and (not (robot-holding-receptical-in-air ?i))
            (robot-gripper-empty))
    )

    (:action pour
        :parameters (?d - drink ?r - receptical)
        :precondition (robot-holding-drink-in-air ?d)
        :effect (is-full ?r)
    )
)