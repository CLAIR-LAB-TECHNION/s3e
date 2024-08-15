(define (domain body-movement)
  (:requirements :strips :typing)
  (:types joint extremity person)
  (:predicates
  	(is-person-joint-bent ?p - person ?j - joint)
  	(is-person-extremity-above-waist ?p - person ?a - extremity)
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

  (:action lift-extremity
	:parameters (?p - person ?a - extremity)
	:precondition ()
	:effect (and
		(is-person-extremity-above-waist ?p ?a)
	)
  )

  (:action drop-extremity
	:parameters (?p - person ?a - extremity)
	:precondition (and
		(is-person-extremity-above-waist ?p ?a)
	)
	:effect (and
		(not (is-person-extremity-above-waist ?p ?a))
	)
  )
)