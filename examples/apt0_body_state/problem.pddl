(define (problem humanoid-apt0)
    (:domain body-movement)
    (:objects 
        left-hand right-hand left-foot right-foot - extremity
        left-elbow right-elbow left-knee right-knee - joint
        girl-with-headphones - person
    )

    (:init)

    (:goal (and
        (is-person-joint-bent girl-with-headphones left-elbow)
    ))
)