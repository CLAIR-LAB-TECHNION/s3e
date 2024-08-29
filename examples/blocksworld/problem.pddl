(define (problem blocksworld)
(:domain n-table-blocks-world)
(:objects
red_block blue_block green_block yellow_block cyan_block purple_block - block
brown-table black-table white-table - table
)
(:init
(robot-gripper-empty)

(on-table red_block black-table)
(clear-on-top red_block)

(on-table blue_block black-table)
(clear-on-top blue_block)

(on-table green_block black-table)
(clear-on-top green_block)

(on-table yellow_block black-table)
(clear-on-top yellow_block)

(on-table cyan_block brown-table)
(clear-on-top cyan_block)

(on-table purple_block brown-table)
(clear-on-top purple_block)


)
(:goal (and
(on-table red_block white-table)
))
)
