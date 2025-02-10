(define (problem pour-drink)

    (:domain groceriesworld)
    (:objects
        green-can - drink
        red-cup - receptical
    )

    (:init
        (robot-gripper-empty)
    )

    (:goal
        (and
            (is-full red-cup)
        )
    )

)