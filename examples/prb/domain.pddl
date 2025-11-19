(define (domain shape-stacking)
    (:requirements :strips :typing :equality)
    (:types
        block sphere - shape
    )
    (:predicates
        (is-directly-on-table ?s - shape)
        (is-clear-on-top ?s - shape)
        (shape-on-top-of-block ?s - shape ?b - block)
    )

    (:action move-from-block-to-table
        :parameters (?s - shape ?b - block)
        :precondition (and 
            (is-clear-on-top ?s)
            (shape-on-top-of-block ?s ?b))
        :effect (and 
            (is-directly-on-table ?s)
            (not (shape-on-top-of-block ?s ?b))
            (is-clear-on-top ?b))
    )

    (:action move-from-table-to-block
        :parameters (?s - shape ?b - block)
        :precondition (and 
            (is-clear-on-top ?s) 
            (is-directly-on-table ?s) 
            (is-clear-on-top ?b)
            (not (= ?s ?b)))
        :effect (and 
            (not (is-directly-on-table ?s))
            (not (is-clear-on-top ?b))
            (shape-on-top-of-block ?s ?b))
    )

    (:action move-from-block-to-block
        :parameters (?s - shape ?b1 ?b2 - block)
        :precondition (and 
            (is-clear-on-top ?s) 
            (shape-on-top-of-block ?s ?b1) 
            (is-clear-on-top ?b2) 
            (not (= ?s ?b2))
            (not (= ?b1 ?b2)))
        :effect (and 
            (not (shape-on-top-of-block ?s ?b1))
            (not (is-clear-on-top ?b2))
            (shape-on-top-of-block ?s ?b2)
            (is-clear-on-top ?b1))
    )
)