(define (problem grocery-bagging)

(:domain item-sorting)
(:objects
milk-carton lemon green-bottle loaf-of-bread red-box-of-cereal red-can-of-soda - item
wood-table black-table white-table - table
dark-wood-container light-wood-container - container)

(:init

(robot-gripper-empty)

(on-table milk-carton wood-table)
(on-table lemon wood-table)
(on-table green-bottle black-table)
(on-table loaf-of-bread black-table)
(on-table red-box-of-cereal black-table)
(on-table red-can-of-soda black-table)

)

(:goal (and
(in-container milk-carton dark-wood-container)
))

)