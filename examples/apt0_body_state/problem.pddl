(define (problem humanoid-apt0)
    (:domain body-movement)
    (:objects 
        left-arm right-arm left-leg right-leg - appendage
        left-elbow right-elbow left-knee right-knee - joint
        girl-with-headphones - person
    )

    (:init)

    (:goal (and
        (is-person-joint-bent girl-with-headphones left-elbow)
    ))
)