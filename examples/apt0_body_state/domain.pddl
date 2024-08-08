(define (domain body-movement)
  (:requirements :strips :typing)
  (:types joint appendage person)
  (:predicates
  	(is-person-joint-bent ?p - person ?j - joint)
  	(is-person-apendage-in-front ?p - person ?a1 ?a2 - appendage)
  	(is-person-mid-air ?p - person)
  )

  (:action bend
	     :parameters (?p - person ?j - joint)
	     :precondition ()
	     :effect (and
	     	(is-person-joint-bent ?p ?j)
	     )
  )

  (:action jump
	  :parameters (?p - person)
	  :precondition ()
	  :effect (and 
	  	(is-person-mid-air ?p)
	  )
  )

  (:action move-in-front
	:parameters (?p - person ?a1 ?a2 - appendage)
	:precondition ()
	:effect (and
		(is-person-apendage-in-front ?p ?a1 ?a2)
		(not (is-person-apendage-in-front ?p ?a2 ?a1))
	)
  )
)