(define (domain shape-stacking)
    (:requirements :strips :typing :equality)
    (:types
        block
    )
    (:predicates
        (is-directly-on-table ?b - block)
        (is-clear-on-top ?b - block)
        (block-on-top-of-block ?b1 ?b2 - block)
    )

    (:action move-from-block-to-table
        :parameters (?b1 ?b2 - block)
        :precondition (and (is-clear-on-top ?b1)(block-on-top-of-block ?b1 ?b2))
        :effect (and (is-directly-on-table ?b1)
            (not (block-on-top-of-block ?b1 ?b2))
            (is-clear-on-top ?b2))
    )

    (:action move-from-table-to-block
        :parameters (?b1 ?b2 - block)
        :precondition (and (is-clear-on-top ?b1) (is-directly-on-table ?b1) (is-clear-on-top ?b2))
        :effect (and (not (is-directly-on-table ?b1))
            (not (is-clear-on-top ?b2))
            (block-on-top-of-block ?b1 ?b2))
    )

    (:action move-from-block-to-block
        :parameters (?b1 ?b2 ?b3 - block)
        :precondition (and (is-clear-on-top ?b1) (block-on-top-of-block ?b1 ?b3) (is-clear-on-top ?b2))
        :effect (and (not (block-on-top-of-block ?b1 ?b3))
            (not (is-clear-on-top ?b2))
            (block-on-top-of-block ?b1 ?b2)
            (is-clear-on-top ?b3))
    )
)