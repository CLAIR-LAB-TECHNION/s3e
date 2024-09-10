(define (domain groceriesworld)
    (:requirements :strips :typing)
    (:types
        item section
    )
    (:predicates
        (in-table-section ?i - item ?s - section)
        (robot-gripper-empty)
        (robot-holding-in-air ?i - item)
    )

    (:action pick-up
        :parameters (?i - item ?s - section)
        :precondition (and (in-table-section ?i ?s)(robot-gripper-empty))
        :effect (and (not (in-table-section ?i ?s))
            (not (robot-gripper-empty))
            (robot-holding-in-air ?i))
    )

    (:action put-down
        :parameters (?i - item ?s - section)
        :precondition (robot-holding-in-air ?i)
        :effect (and (not (robot-holding-in-air ?i))
            (robot-gripper-empty)
            (in-table-section ?i ?s))
    )
)