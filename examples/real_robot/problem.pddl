(define (problem item-sorting)

    (:domain groceriesworld)
    (:objects
        green-mug red-can mineral-water-bottle spray-bottle - item
        blue white - section
    )

    (:init
        (robot-gripper-empty)

        (in-table-section green-mug white)
        (in-table-section mineral-water-bottle white)
        (in-table-section red-can blue)
        (in-table-section spray-bottle blue)
    )

    (:goal
        (and
            (in-table-section green-mug white)
            (in-table-section red-can white)
            (in-table-section spray-bottle white)
            (in-table-section mineral-water-bottle blue)
        )
    )

)