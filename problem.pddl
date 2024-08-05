(define (problem grocery-bagging)

(:domain item-sorting)
(:objects
milk-carton lemon green-bottle loaf-of-bread red-box-of-cereal red-can-of-soda - item
dark-wood-container light-wood-container - container)

(:init

(gripper-empty)

(on-table milk-carton)
(on-table lemon)
(on-table green-bottle)
(on-table loaf-of-bread)
(on-table red-box-of-cereal)
(on-table red-can-of-soda)

)

(:goal (and
(in-container milk-carton dark-wood-container)
))

)