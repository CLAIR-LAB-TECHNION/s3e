(define (problem item-sorting)

    (:domain groceriesworld)
    (:objects
        red-cup green-can spray-bottle - item
        blue white - section
    )

    (:init
        (robot-gripper-empty)

        (in-table-section red-cup white)
        (in-table-section green-can blue)
        (in-table-section spray-bottle blue)
    )

    (:goal
        (and
            (in-table-section red-cup white)
            (in-table-section green-can white)
            (in-table-section spray-bottle white)
        )
    )

)