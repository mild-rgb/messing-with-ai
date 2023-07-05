"""
two sentences do not mean the same thing if they contradict each other - 
it is warm outside vs it is cold outside 

for the sake of the exercise, i assume that cold and fridig mean the same thing. while they do not, anything cold could plausibly be described as frigid. 


all of the terms used to test the model must be contained in the training dataset. 
"""

from transformers import pipeline 

testline = pipeline("text-classification", "transformers-course/classifier1") #change the path to where your classifier is 


def sum_up(input_tuple, target): #target should either be LABEL_0 or LABEL_1
	certainty = 0
	score = 0
	for entry in input_tuple:
		result = testline(entry)
		certainty = certainty + result[0]['score']
		#print(result) #this is for debugging purposes
		if(result[0]['label'] == target):
			score += 1
	print("score")		
	print(score/7)
	print("certainty")
	print(certainty/7)


#these are all simple sentences with simple synonym substitutions and should all return label 1. 
test_sentences_0 =  iter((
 "the doctor sat on the beach""the physician sat on the beach",
 "the wind blew through the door""the wind blew through the entrance",
 "i warmed up the bread in the oven""i heated up the bread in the oven", 
 "hugo is working together with amy""hugo is collaborating with amy",
 "the sun is rising in a few hours""the sun is coming up in a few hours",
 "the oxen here are sturdy""the oxen here are strong",
 "the storm drenched us""the storm soaked us"
))




sum_up(test_sentences_0, "LABEL_1")
print("NEXT RUN BELOW")
#these sentences are all contradictory and should return label 0
test_sentences_1 = iter((
"the cat meowed" "the cat was silent",
"the hawk flew over the valley""the hawk sat in the valley",
"the sun shone""the sun did not shine", 
"it was dark in the forest""it was light in the forest",
"the computer shut down""the computer stayed on",
"the storm drenched us" "the storm did not drench us",
"the sea was rough that day" "the sea was calm that day"
))
sum_up(test_sentences_1, "LABEL_0")






