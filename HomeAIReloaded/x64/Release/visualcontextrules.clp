(deftemplate visualdetect
				 (slot type (default object))
				 (multislot rectangle)
				 (slot ontology)
				 (slot at)
				 (slot timestamp)
				 )
(deffunction count ($?arg) 
	(length $?arg))

(deffunction cyclicCallback ($?a) 

	(printout t "Total Face Facts=" (count (find-all-facts ((?f visualdetect)) (= (str-compare ?f:ontology "human") 0) )) crlf)
	(printout t "Total Text Facts=" (count (find-all-facts ((?f visualdetect)) (= (str-compare ?f:ontology "text") 0) )) crlf)
	(printout t "Total Contour Facts=" (count (find-all-facts ((?f visualdetect)) (= (str-compare ?f:ontology "contour") 0) )) crlf)
	
	)
	

